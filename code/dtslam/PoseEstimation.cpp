/*
 * PoseEstimation.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "PoseEstimation.h"
#include <opencv2/calib3d.hpp>
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <ceres/ceres.h>
#include "cvutils.h"
#include "CeresUtils.h"
#include "Profiler.h"
#include "flags.h"

namespace dtslam {

void PoseEstimator::init(const CameraModel *camera, float outlierPixelThreshold)
{
	mCamera = camera;
	mOutlierPixelThreshold = outlierPixelThreshold;

	mMinRansacIterations = FLAGS_PoseMinRansacIterations;
	mMaxRansacIterations = FLAGS_PoseMaxRansacIterations;

	mRefineOnlyInliers = false;

	mEssentialRefiner.reset(new EssentialRefiner());
	mEssentialRefiner->setCamera(mCamera);
	mEssentialRefiner->setOutlierThreshold(mOutlierPixelThreshold);

	mRotationRefiner.reset(new RotationRefiner());
	mRotationRefiner->setCamera(mCamera);
	mRotationRefiner->setOutlierThreshold(mOutlierPixelThreshold);

	mPoseRefiner.reset(new PnPRefiner());
	mPoseRefiner->setCamera(mCamera);
	mPoseRefiner->setOutlierThreshold(mOutlierPixelThreshold);
}

void PoseEstimator::fitModels(std::vector<FeatureMatch> &matches)
{
	//Count 3D and 2D
	int count3D=0;
	int count2D=0;
	for (auto &match : matches)
	{
		if (match.measurement.getFeature().is3D())
			count3D++;
		else
			count2D++;
	}

	mReprojectionErrors.resize(matches.size());
	mIsStable.resize(matches.size(), false);

	if (count3D >= FLAGS_TrackerMinMatchCount)
	{
		fitPnP(matches);
	}
	else if (count2D >= FLAGS_TrackerMinMatchCount)
	{
		fitEssential(matches);
	}
	else
	{
		DTSLAM_LOG << "Not enough features found!\n";
		mPoseType = EPoseEstimationType::Invalid;
	}
}

SlamKeyFrame *PoseEstimator::findEssentialReferenceFrame(std::vector<FeatureMatch> &matches2D)
{
	SlamKeyFrame *referenceFrame = NULL;

	std::unordered_map<SlamKeyFrame *, int> frameMatchCount;
	for (auto &match : matches2D)
	{
		for (auto &mPtr : match.sourceMeasurement->getFeature().getMeasurements())
		{
			SlamKeyFrame *frame = &mPtr->getKeyFrame();
			auto it = frameMatchCount.insert(std::make_pair(frame, 0));
			it.first->second++;
		}
	}
	int maxCount = 0;
	for (auto &it : frameMatchCount)
	{
		if (it.second > maxCount) //Select frame with highest match count
		{
			maxCount = it.second;
			referenceFrame = it.first;
		}
		else if (it.second == maxCount && it.first->getTimestamp() > referenceFrame->getTimestamp()) //If tied, select newest frame
		{
			referenceFrame = it.first;
		}
	}

	//Replace sourceMeasurement to match reference frame
	for (auto &match : matches2D)
	{
		if (&match.sourceMeasurement->getKeyFrame() != referenceFrame)
		{
			for (auto &mPtr : match.sourceMeasurement->getFeature().getMeasurements())
			{
				SlamKeyFrame *frame = &mPtr->getKeyFrame();
				if (frame == referenceFrame)
				{
					if (mPtr->getPositionCount() == 1) //Only set as source measurement if it is a unique match
						match.sourceMeasurement = mPtr.get();
					break;
				}
			}
		}
	}

	return referenceFrame;
}

void PoseEstimator::fitEssential(std::vector<FeatureMatch> &matches)
{
	int matchCount = matches.size();

	//Select a reference frame
	mEssentialReferenceFrame = findEssentialReferenceFrame(matches);
	assert(mEssentialReferenceFrame);

	//Select matches from the reference frame
	std::vector<FeatureMatch> referenceFrameMatches;
	for(auto &match : matches)
	{
		if(&match.sourceMeasurement->getKeyFrame() == mEssentialReferenceFrame)
			referenceFrameMatches.push_back(match);
	}

	//Prepare vectors
	std::unordered_map<SlamKeyFrame*,cv::Matx33f> refPosesMap;

	//The minimum number of acceptable inliers to stop ransac
	const int acceptableCount = (int)(matchCount*0.90f);

	static int frameId=0;
	frameId++;

	//Try rotation model
	cv::Vec3f dominantFrameCenter = mEssentialReferenceFrame->getPose().getCenter();
	cv::Matx33f pureR;
	{
		ProfileSection ss("rotationRansac");

		Rotation3DRansac rransac;

		int rotationAcceptableCount = acceptableCount;

		rransac.setParams(mOutlierPixelThreshold, mMinRansacIterations, mMaxRansacIterations, rotationAcceptableCount);
		rransac.setData(&referenceFrameMatches);
		rransac.doRansac();
		if (rransac.getBestIterationData())
		{
			pureR = rransac.getBestModel();
			mPureRotationInlierCount = rransac.getBestInlierCount();
		}
		else
		{
			//Ransac failed
			DTSLAM_LOG << "Rotation ransac failed!\n";
			pureR = cv::Matx33f::eye();
			mPureRotationInlierCount = 0;
		}

		DTSLAM_LOG << "Rotation estimation RANSAC: iterations=" << rransac.getIterationsMade() << ", inliers=" << rransac.getBestInlierCount() << "\n";
	}

	{
		ProfileSection ss("rotationRefine");

		//mRotationRefiner->setOutlierThreshold(2*mOutlierPixelThreshold);
		mRotationRefiner->refineRotation(matches, dominantFrameCenter, pureR, mPureRotationInlierCount, mPureRotationErrors);
	}
	DTSLAM_LOG << "Rotation refinement ceres: inliers=" << mPureRotationInlierCount << "\n";

	//Skip essential
	bool isPureRotation;
	bool skipEssential;
	if (FLAGS_DisableRegions && mEssentialReferenceFrame->getRegion()->getFirstTriangulationFrame()) 	//Force rotation model after initialization
		skipEssential = true;
	else if (mPureRotationInlierCount >= (int)(0.9f*matches.size()))
		skipEssential = true;
	else
		skipEssential = false;

	//Try an essential matrix
	FullPose3D essentialPose;

	if (skipEssential)
	{
		//Skip essential model estimation
		mInlierCount = mPureRotationInlierCount;
		mReprojectionErrors = mPureRotationErrors;
		isPureRotation = true;

		//Count tracks of valid length
		mIsStable.resize(matchCount, false);
		for (int i = 0, end = matchCount; i<end; ++i)
		{
			auto &match = matches[i];
			if (match.trackLength > FLAGS_PoseMinTrackLength)
			{
				if (mReprojectionErrors[i].isInlier)
				{
					mIsStable[i] = true;
				}
			}
		}
	}
	else
	{
		{
			ProfileSection ss("essentialRansac");

			int essentialAcceptableCount = acceptableCount;

			EssentialRansac eransac;
			eransac.setParams(mOutlierPixelThreshold, mMinRansacIterations, mMaxRansacIterations, essentialAcceptableCount);
			eransac.setData(mCamera, &referenceFrameMatches, &matches);

			eransac.doRansac();
			if (eransac.getBestIterationData())
			{
				//DTSLAM_LOG << "Essential estimation RANSAC: iterations=" << eransac.getIterationsMade() << ", inliers=" << eransac.getBestInlierCount() << "\n";
				//DTSLAM_LOG << "Essential RANSAC pose: R=" << essentialPose.getRotationRef() << ", t=" << essentialPose.getTranslationRef() << "\n";
				//DTSLAM_LOG << "Best model: " << eransac.getBestModel() << "\n";

				mInlierCount = eransac.getBestInlierCount();
				essentialPose = eransac.getBestModel().pose;
				mReprojectionErrors = std::move(eransac.getBestIterationData()->reprojectionErrors);
				//points4 = std::move(eransac.getBestIterationData()->points4);

				const int inlierPercentage = (int)(100 * eransac.getBestInlierCount() / (float)matchCount);
				DTSLAM_LOG << "Essential estimation RANSAC: iterations=" << eransac.getIterationsMade() << ", inliers=" << eransac.getBestInlierCount() << " (" << inlierPercentage << "%)\n";
				//DTSLAM_LOG << "Essential RANSAC pose: R=" << (cv::Mat)essentialPose.getRotationRef() << ", t=" << (cv::Mat)essentialPose.getTranslationRef() << "\n";
			}
			else
			{
				//Ransac failed, use any pose
				DTSLAM_LOG << "Essential ransac failed!\n";
				essentialPose = mEssentialReferenceFrame->getPose();
				mInlierCount = 0;
				mReprojectionErrors.resize(matchCount);
			}
		}

		//Triangulate (for display in map only)
		//std::vector<cv::Vec4f> &points4 = mData->essentialTriangulations;
		//triangulatePoints(usefulMatches, essentialPose, points4);


		//Refine with ceres
		{
		ProfileSection ss("essentialRefine");

		mEssentialRefiner->refineEssential(matches, essentialPose.getRotationRef(), essentialPose.getTranslationRef(), mInlierCount, mReprojectionErrors);
		}

		//DTSLAM_LOG << "Recovered pose: R=" << (cv::Mat)essentialPose.getRotationRef() << ", t=" << (cv::Mat)essentialPose.getTranslationRef() << "\n";

		///////////////////////////////////////////////////////////////
		// Select a model

		//Count tracks of valid length
		int validEssentialTrackCount = 0;
		int validPureRotationTrackCount=0;
		mIsStable.resize(matchCount, false);
		for (int i = 0, end = matchCount; i<end; ++i)
		{
			auto &match= matches[i];
			if(match.trackLength > FLAGS_PoseMinTrackLength)
			{
				if(mReprojectionErrors[i].isInlier)
				{
					validEssentialTrackCount++;
					mIsStable[i] = true;
				}
				if(mPureRotationErrors[i].isInlier)
					validPureRotationTrackCount++;
			}
		}

		//Compare models
		isPureRotation = false;
		DTSLAM_LOG << "Valid essential tracks = " << validEssentialTrackCount << ", valid rotation tracks = " << validPureRotationTrackCount;
		if (validEssentialTrackCount > 20 || validPureRotationTrackCount > 20)
		{
			const float validRatio = static_cast<float>(validEssentialTrackCount - validPureRotationTrackCount) / validEssentialTrackCount;
			isPureRotation = (validRatio < FLAGS_PoseStableRatioThreshold);
			DTSLAM_LOG << ", ratio = " << validRatio;

		}
		else
		{
			DTSLAM_LOG << ", not enough for ratio";

		}
		DTSLAM_LOG << " ->" << (isPureRotation ? "rotation" : "essential") << "\n";
	}

	if(isPureRotation)
	{
		//Assume pure rotation
		mPoseType = EPoseEstimationType::PureRotation;
		mPose.set(pureR, -pureR*dominantFrameCenter);
	}
	else
	{
		//Assume full model
		//DTSLAM_LOG << "Assuming full motion.\n";
		mPoseType = EPoseEstimationType::Essential;
		mPose = essentialPose;
	}
	DTSLAM_LOG << "FitEssential: Assuming " << ((mPoseType==EPoseEstimationType::PureRotation)?"pure rotation":"full motion") << ", "
			<< "R=" << (cv::Mat)mPose.getRotationRef() << ", t=" << mPose.getTranslationRef() << "\n";
}

void PoseEstimator::fitPnP(std::vector<FeatureMatch> &matches)
{
	const int matchCount = matches.size();

	//The minimum number of acceptable inliers to stop ransac
	const int acceptableCount = (int)(matchCount*0.70f);

	//Ransac
	std::unique_ptr<PnPIterationData> pnpIterationData;

	{
		ProfileSection ss("pnpRansac");

		PnPRansac pransac;
		pransac.setParams(mOutlierPixelThreshold, mMinRansacIterations, mMaxRansacIterations, acceptableCount);
		pransac.setData(&matches, mCamera);
		pransac.doRansac();
		mPose = pransac.getBestModel();
		pnpIterationData = std::move(pransac.getBestIterationData());

		const int inlierPercentage = (int)(100*pransac.getBestInlierCount()/ (float)matchCount);
		DTSLAM_LOG << "Pose estimation RANSAC: iterations=" << pransac.getIterationsMade() << ", inliers=" << pransac.getBestInlierCount() << " (" << inlierPercentage << "%)\n";
	}

	//Refine with ceres
	{
		ProfileSection ss("pnpRefine");

		mPoseRefiner->refinePose(matches, mPose.getRotationRef(), mPose.getTranslationRef(), mInlierCount, mReprojectionErrors);

		const float inlierPercentage = 100*mInlierCount / (float)(matchCount);
		DTSLAM_LOG << "Pose estimation refinement (pixels): inliers=" << mInlierCount << " (" << inlierPercentage << "%)\n";
	}

	//DTSLAM_LOG << "PoseR=\n" << (cv::Mat)pose.getRotationRef() << "\n";
	//DTSLAM_LOG << "PoseT=\n" << pose.getTranslationRef() << "\n";

	//Triangulate (for display in map only)
	//std::vector<cv::Vec4f> &points4 = mData->essentialTriangulations;
	//triangulatePoints(matches2D, pose, points4);

	//Mark stable matches
	mIsStable.resize(matchCount, false);
	for(int i=0; i<matchCount; ++i)
	{
		mIsStable[i] = mReprojectionErrors[i].isInlier && (matches[i].trackLength > FLAGS_PoseMinTrackLength);
	}

	//Pose type
	mPoseType = EPoseEstimationType::FullPose;
}

void PoseEstimator::refinePose(std::vector<FeatureMatch> &matches,
		const EPoseEstimationType poseType, FullPose3D &pose)
{
	ProfileSection ss("refinePose");

	//Count 3D and 2D
	int matchCount = matches.size();
	int count3D = 0;
	int count2D = 0;
	for (auto &match : matches)
	{
		if (match.measurement.getFeature().is3D())
			count3D++;
		else
			count2D++;
	}

	//Refine
	if (poseType == EPoseEstimationType::FullPose)
	{
		mPureRotationErrors.resize(matchCount);		

		mPoseRefiner->refinePose(matches, pose.getRotationRef(), pose.getTranslationRef(), mInlierCount, mReprojectionErrors);
	}
	else
	{
		mEssentialReferenceFrame = findEssentialReferenceFrame(matches);

		if (poseType == EPoseEstimationType::Essential)
		{
			mPureRotationErrors.resize(matchCount);

			mEssentialRefiner->refineEssential(matches, pose.getRotationRef(), pose.getTranslationRef(), mInlierCount, mReprojectionErrors);
		}
		else
		{
			mIsStable.resize(matchCount, false);

			cv::Matx33f R = pose.getRotationRef();
			cv::Vec3f center = pose.getCenter();
			mRotationRefiner->refineRotation(matches, center, R, mInlierCount, mReprojectionErrors);
			pose.set(R, -R*center);

			mPureRotationErrors = mReprojectionErrors;
		}
	}

	//Triangulate (for display in map only)
	//std::vector<cv::Vec4f> &points4 = mData->essentialTriangulations;
	//triangulatePoints(matches2D, pose, points4);

	DTSLAM_LOG << "Recovered pose: R=" << (cv::Mat)pose.getRotationRef() << ", t=" << (cv::Mat)pose.getTranslationRef() << "\n";
	mPose = pose;
	mPoseType = poseType;
}

} /* namespace dtslam */
