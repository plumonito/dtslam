/*
 * SlamMapExpander.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "SlamMapExpander.h"
#include "SlamMap.h"
#include "SlamKeyFrame.h"
#include "PoseTracker.h" //For the flags
#include "SlamSystem.h"

#include "flags.h"

namespace dtslam
{

	bool SlamMapExpander::init(const CameraModel *camera, SlamSystem *slam)
{
	mCamera = camera;
	mSlam = slam;

	mMinTriangulationAngle = (float)(FLAGS_ExpanderMinTriangulationAngle*M_PI/180);
	DTSLAM_LOG << "SlamMapExpander: MinTriangulationAngle=" << mMinTriangulationAngle << " (rad)\n";

	mFeatureCoverageMaskScale = mCamera->getImageSize().width / kFeatureCoverageCells;
	mFeatureCoverageMask = cv::Mat1b(mCamera->getImageSize().height / mFeatureCoverageMaskScale, mCamera->getImageSize().width / mFeatureCoverageMaskScale);

	mMatcher.reset(new FeatureMatcher());
	mMatcher->setCamera(mCamera);
	mMatcher->setSearchDistance(2*FLAGS_MatcherPixelSearchDistance);
	//mMatcher->setNonMaximaPixelSize(0);
	//mMatcher->setBestScorePercentThreshold(1000);

	mPoseEstimator.reset(new PoseEstimator());
	mPoseEstimator->init(mCamera, 3);

	return true;
}

bool SlamMapExpander::checkFrame(std::unique_ptr<CheckData> data)
{
	mData = std::move(data);
	mRefinedMatches.clear();
	mRefinedMatchesReadyForTriangulation.clear();
	mRefinedMatchesInliers.clear();
	mReadyForTriangulationCount = 0;

	prepareFeaturesInView();

	mFeatureCoverageMask.setTo(ECellEmpty);

	bool shouldAdd;
	if (checkTrackedInfo())
	{
		mStatus = ESlamMapExpanderStatus::CheckingFrame;
		shouldAdd = checkCompleteFrame();
	}
	else
		shouldAdd = false;

	mStatus = ESlamMapExpanderStatus::Inactive;
	return shouldAdd;
}

bool SlamMapExpander::checkTrackedInfo()
{
	bool shouldAdd = mData->forceAdd;
	int matchCount = mData->trackedFeatures.size();

	//Check if there is enough parallax to triangulate
	mTrackerMatchesReadyForTriangulationCount = 0;
	mTrackerMatchAngles.resize(matchCount);
	for(int i=0; i<matchCount; ++i)
	{
		FeatureMatch &match = mData->trackedFeatures[i];
		mTrackerMatchAngles[i] = getMatchTriangulationAngle(match);

		//Check angle
		if (mTrackerMatchAngles[i] >= mMinTriangulationAngle)
			mTrackerMatchesReadyForTriangulationCount++;
	}
	if(mTrackerMatchesReadyForTriangulationCount>0)
		DTSLAM_LOG << "Tracker matches ready for triangulation: " << mTrackerMatchesReadyForTriangulationCount << ".\n";
	if(mTrackerMatchesReadyForTriangulationCount > FLAGS_ExpanderMinNewTriangulationsForKeyFrame)
	{
		DTSLAM_LOG << "Enough matches ready for triangulation to add a new key frame.\n";
		shouldAdd = true;
	}

	//Check that there are new features to add
	markTrackerMatches();

	//Do initial check with already matched features
	if (!shouldAdd && checkFeatureCoverageRatio())
	{
		DTSLAM_LOG << "Initial feature coverage ratio warrants a new key frame\n";
		shouldAdd = true;
	}

	return shouldAdd;
}

bool SlamMapExpander::checkCompleteFrame()
{
	findExtraMatches();

	//Check triangulation
	//Triangulation must go first so that mRefinedMatchesReadyForTriangulation is always propagated
	mReadyForTriangulationCount = checkRefinedMatchesForTriangulations();
	if (mReadyForTriangulationCount>0)
		DTSLAM_LOG << "Refined matches ready for triangulation: " << mReadyForTriangulationCount << ".\n";
	if (mReadyForTriangulationCount > FLAGS_ExpanderMinNewTriangulationsForKeyFrame)
	{
		DTSLAM_LOG << "Adding keyframe because of new triangulations.\n";
		return true;
	}

	//Change essential model if not enough triangulations found
	if (mData->poseType == EPoseEstimationType::Essential)
	{
		DTSLAM_LOG << "Expander: Essential model changed back to rotation model because not enough features can be triangulated.\n";
		changeEssentialToRotation();
	}

	//Check coverage ratio
	if (checkFeatureCoverageRatio())
	{
		DTSLAM_LOG << "Adding keyframe because of coverage ratio.\n";
		return true;
	}

	return false;
}

int SlamMapExpander::checkRefinedMatchesForTriangulations()
{
	if (mData->poseType == EPoseEstimationType::FullPose)
		DTSLAM_LOG << "Checking for triangulations: all frames\n";
	else
		DTSLAM_LOG << "Checking for triangulations: only essential reference frame\n";

	int count = 0;
	mRefinedMatchesReadyForTriangulation.resize(mRefinedMatches.size());
	for (int i = 0, end = mRefinedMatches.size(); i != end; ++i)
	{
		if (!mRefinedMatchesInliers[i])
			mRefinedMatchesReadyForTriangulation[i] = false;
		else
		{
			auto &match = mRefinedMatches[i];
			SlamFeature &feature = match.measurement.getFeature();
	
			float angle;
			if (mData->poseType == EPoseEstimationType::FullPose)
				angle = getMatchTriangulationAngle(match);
			else
				angle = getMatchTriangulationAngleForEssential(match, mData->essentialReferenceFrame);

			if (angle >= mMinTriangulationAngle)
			{
				count++;
				mRefinedMatchesReadyForTriangulation[i] = true;
			}
			else
			{
				mRefinedMatchesReadyForTriangulation[i] = false;
			}
		}
	}
	return count;
}

void SlamMapExpander::changeEssentialToRotation()
{
	mData->poseType = EPoseEstimationType::PureRotation;
	
	FullPose3D pose(mData->essentialReferenceFrame->getPose());
	mPoseEstimator->refinePose(mRefinedMatches, EPoseEstimationType::PureRotation, pose);

	mData->frame->setPose(std::unique_ptr<Pose3D>(new RelativeRotationPose3D(&mData->essentialReferenceFrame->getPose(), pose.getRotationRef())));
	
	mRefinedMatchesReadyForTriangulation.clear();
	mRefinedMatchesReadyForTriangulation.resize(mRefinedMatches.size(), false);
}

float SlamMapExpander::getMatchTriangulationAngle(const FeatureMatch &match)
{
	if (match.measurement.getPositions().size() > 1)
	{
		return -1.0f;
	}
	else
	{
		SlamFeature &feature = match.measurement.getFeature();

		if (feature.is3D())
		{
			return feature.getMinTriangulationAngle(match.measurement);
		}
		else
		{
			SlamFeatureMeasurement *m2;
			float angle;
			feature.getMeasurementsForTriangulation(match.measurement, m2, angle);
			return angle;
		}
	}
}

float SlamMapExpander::getMatchTriangulationAngleForEssential(const FeatureMatch &match, SlamKeyFrame *essentialReferenceFrame)
{
	if (match.measurement.getPositions().size() > 1)
	{
		return -1.0f;
	}

	SlamFeature &feature = match.measurement.getFeature();
	SlamFeatureMeasurement *mRef = NULL;

	for (auto &mPtr : feature.getMeasurements())
	{
		if (&mPtr->getKeyFrame() == essentialReferenceFrame)
		{
			mRef = mPtr.get();
			break;
		}
	}

	if (!mRef || mRef->getPositionCount() > 1)
		return -1.0f;
	else
		return SlamFeature::GetTriangulationAngle(*mRef, match.measurement);
}

void SlamMapExpander::prepareFeaturesInView()
{
	mFeaturesInView.clear();
	mRegion->getFeaturesInView(*mData->frame,false,std::unordered_set<SlamFeature*>(),mFeaturesInView);

	std::vector<int> featuresInViewCount;
	for(int octave=0, end=mFeaturesInView.size(); octave!=end; ++octave)
		featuresInViewCount.push_back(mFeaturesInView[octave].size());
	DTSLAM_LOG << "Expander: features in view=" << featuresInViewCount << "\n";
}

void SlamMapExpander::markTrackerMatches()
{
	//Mark features in view
	for(int octave=0; octave<(int)mFeaturesInView.size(); ++octave)
	{
		auto &features = mFeaturesInView[octave];
		for(auto &projection : features)
		{
			if(projection.getType() == EProjectionType::EpipolarLine)
				continue;
			assert(projection.getPointData().positions.size() == 1);
			markPointAsCovered(projection.getPointData().positions[0], octave);
		}
	}

	//Mark features from tracker
	for(auto &match : mData->trackedFeatures)
	{
		for(auto &pos : match.measurement.getPositions())
			markPointAsCovered(pos, match.measurement.getOctave());
	}
}

void SlamMapExpander::markPointAsCovered(const cv::Point2i &p, const int octave)
{
	const int scale = 1<<octave;

	cv::Point2i topLeft((p.x - scale*kFeatureCoverageSize) / mFeatureCoverageMaskScale, (p.y - scale*kFeatureCoverageSize) / mFeatureCoverageMaskScale);
	if(topLeft.x < 0)
		topLeft.x = 0;
	if(topLeft.y < 0)
		topLeft.y = 0;
	cv::Point2i bottomRight((p.x + scale*kFeatureCoverageSize) / mFeatureCoverageMaskScale, (p.y + scale*kFeatureCoverageSize) / mFeatureCoverageMaskScale);
	if(bottomRight.x >= mFeatureCoverageMask.cols)
		bottomRight.x = mFeatureCoverageMask.cols-1;
	if(bottomRight.y >= mFeatureCoverageMask.rows)
		bottomRight.y = mFeatureCoverageMask.rows-1;

	for(int v=topLeft.y; v<=bottomRight.y; ++v)
	{
		auto row = mFeatureCoverageMask[v];
		for(int u=topLeft.x; u<=bottomRight.x; ++u)
		{
			row[u] = ECellCoveredByOld;
		}
	}
}

bool SlamMapExpander::checkFeatureCoverageRatio()
{
	int coveredByOldCount=0;
	int coveredByNewCount=0;

	//Count cells already covered
	for(int v=0; v<mFeatureCoverageMask.rows; ++v)
	{
		auto row = mFeatureCoverageMask[v];
		for(int u=0; u<mFeatureCoverageMask.cols; ++u)
		{
			if(row[u] == ECellCoveredByOld)
				coveredByOldCount++;
			else if(row[u] == ECellCoveredByNew)
				coveredByNewCount++;
		}
	}

	//Count new cells covered by new key points
	for(int octave=0; octave<mData->frame->getPyramid().getOctaveCount(); ++octave)
	{
		for(auto it=mData->frame->getKeyPoints(octave).begin(), end=mData->frame->getKeyPoints(octave).end(); it!=end; ++it)
		{
			const KeyPointData &keyPoint = *it;
			const int scale = 1<<octave;
			const cv::Point2f p = keyPoint.position;

			cv::Point2i topLeft((int)((p.x - scale*kFeatureCoverageSize) / mFeatureCoverageMaskScale), (int)((p.y - scale*kFeatureCoverageSize) / mFeatureCoverageMaskScale));
			if(topLeft.x < 0)
				topLeft.x = 0;
			if(topLeft.y < 0)
				topLeft.y = 0;
			cv::Point2i bottomRight((int)((p.x + scale*kFeatureCoverageSize) / mFeatureCoverageMaskScale), (int)((p.y + scale*kFeatureCoverageSize) / mFeatureCoverageMaskScale));
			if(bottomRight.x >= mFeatureCoverageMask.cols)
				bottomRight.x = mFeatureCoverageMask.cols-1;
			if(bottomRight.y >= mFeatureCoverageMask.rows)
				bottomRight.y = mFeatureCoverageMask.rows-1;

			for(int v=topLeft.y; v<=bottomRight.y; ++v)
			{
				auto row = mFeatureCoverageMask[v];
				for(int u=topLeft.x; u<=bottomRight.x; ++u)
				{
					if(row[u]==ECellEmpty)
					{
						row[u] = ECellCoveredByNew;
						coveredByNewCount++;
					}
				}
			}
		}
	}


	float ratio;
	if(coveredByOldCount==0)
		ratio = (float)coveredByNewCount;
	else
		ratio = static_cast<float>(coveredByNewCount) / coveredByOldCount;

	bool result = ratio > FLAGS_ExpanderNewCoverageRatioForKeyFrame;

	DTSLAM_LOG << "Feature coverage ratio: " << ratio << (result ? " (add new!)" : "") << "\n";

	return result;
}

void SlamMapExpander::findExtraMatches()
{
	ProfileSection s("findExtraMatches");

	DTSLAM_LOG << "Expander: finding extra matches\n";

	//Build a set of already matched features
	std::unordered_set<SlamFeature *> alreadyMatched;
	for(auto &match : mData->trackedFeatures)
		alreadyMatched.insert(&match.measurement.getFeature());

	//Create vector with all found features
	std::vector<FeatureMatch> allMatches;

	//Add tracked features
	allMatches.insert(allMatches.end(), mData->trackedFeatures.begin(), mData->trackedFeatures.end());

	//Refine pose iteratively
	FullPose3D refinedPose(mData->frame->getPose());
	cv::Vec3f refinedPoseCenter = refinedPose.getCenter();

	mPoseEstimator->setRefineOnlyInliers(false);

	int maxOctave = mFeaturesInView.size()-1;
	bool poseRefined = false;

	for(int octave = maxOctave; octave>=0; --octave)
	{
		DTSLAM_LOG << "\nRefining pose, octave " << octave << "\n";

		std::vector<FeatureProjectionInfo> featuresToMatch;
		for(auto &projection : mFeaturesInView[octave])
		{
			SlamFeature &feature = projection.getFeature();

			if(alreadyMatched.find(&feature) != alreadyMatched.end())
				continue;

			if(octave==maxOctave)
			{
				featuresToMatch.push_back(projection);
			}
			else
			{
				//Update projection
				FeatureProjectionInfo updated;

				if(feature.is3D())
					updated = SlamRegion::Project3DFeature(refinedPose, refinedPoseCenter, *mCamera, mData->frame->getPyramid().getOctaveCount(), feature);
				else
					updated = SlamRegion::Project2DFeature(refinedPose, refinedPoseCenter, *mCamera, projection.getSourceMeasurement());
				if(updated.getType() != EProjectionType::Invalid)
				{
					featuresToMatch.push_back(updated);
				}
			}
		}

		if (featuresToMatch.empty())
		{
			DTSLAM_LOG << "No new features to search for.\n";
			continue;
		}

		//Match
		mMatcher->clearResults();
		mMatcher->setFrame(mData->frame.get());
		mMatcher->setFramePose(refinedPose);
		{
			//ProfileSection s("matching");
			mMatcher->findMatches(featuresToMatch);
		}

		//Add matches to list
		allMatches.insert(allMatches.end(), mMatcher->getMatches().begin(), mMatcher->getMatches().end());

		//Refine pose
		mPoseEstimator->refinePose(allMatches, mData->poseType, refinedPose);
		refinedPoseCenter = refinedPose.getCenter();
		poseRefined = true;
	}

	if (!poseRefined)
	{
		//Refine pose
		mPoseEstimator->refinePose(allMatches, mData->poseType, refinedPose);
		refinedPoseCenter = refinedPose.getCenter();
	}

	//Re-estimated reference frame
	mData->essentialReferenceFrame = mPoseEstimator->getEssentialReferenceFrame();

	//Copy pose
	if (mData->poseType == EPoseEstimationType::PureRotation)
		mData->frame->setPose(std::unique_ptr<Pose3D>(new RelativeRotationPose3D(&mData->essentialReferenceFrame->getPose(), refinedPose.getRotationRef())));
	else
		mData->frame->setPose(std::unique_ptr<Pose3D>(new FullPose3D(refinedPose)));
	
	//Copy refined matches
	int ambiguousCount=0;
	int outlierCount=0;
	int matchCount=allMatches.size();
	for(int i=0,end=matchCount; i!=end; ++i)
	{
		FeatureMatch &match = allMatches[i];
		MatchReprojectionErrors &errors = mPoseEstimator->getReprojectionErrors()[i];
		if(!errors.isInlier)
			outlierCount++;

		if(match.measurement.getPositions().size() > 1)
			ambiguousCount++;

		mRefinedMatches.push_back(match);
		mRefinedMatchesInliers.push_back(errors.isInlier);
		//Note: marking the match as outlier prevents it from being triangulated. This is good!
	}
	DTSLAM_LOG << "From all matches (" << matchCount << ") found: outliers=" << outlierCount << ", ambiguous=" << ambiguousCount << ", refinedCount=" << mRefinedMatches.size() << ".\n";

	//Mark in the feature coverage
	for(auto &match : mRefinedMatches)
	{
		for(auto &pos : match.measurement.getPositions())
			markPointAsCovered(pos, match.measurement.getOctave());
	}
}

SlamKeyFrame *SlamMapExpander::addKeyFrame()
{
	ProfileSection s("addKeyFrame");
	mStatus = ESlamMapExpanderStatus::AddingFrame;

	DTSLAM_LOG << "---------------------\n"
				<< "Adding new key frame, ID=" << mData->frame->getTimestamp() << "!\n"
				<< "---------------------\n";

	/////////////////////////////////////////////
	//Check if we need to start a new region because of scale mismatch
	if (mRegion->getFirstTriangulationFrame() && mData->poseType == EPoseEstimationType::Essential)
	{
		DTSLAM_LOG << "\n-------------------------\nBeginning new region\n-------------------------\n";

		//Create new active region
		mRegion = mSlam->getMap().createRegion();

		//Duplicate reference key frame
		SlamKeyFrame *referenceFrameOld = mData->essentialReferenceFrame;
		mRegion->setPreviousRegionSourceFrame(referenceFrameOld);

		std::unique_ptr<SlamKeyFrame> referenceFrameNew = referenceFrameOld->copyWithoutFeatures();
		referenceFrameNew->setPose(std::unique_ptr<Pose3D>(new FullPose3D())); //Identity

		//Duplicate features
		std::unordered_map<SlamFeature*, SlamFeature*> featureMap;
		mData->frame->getMeasurements().clear();
		for (auto &mPtr : referenceFrameOld->getMeasurements())
		{
			auto &m = *mPtr;
			SlamFeature *newFeature = mRegion->createFeature2D(*referenceFrameNew, m.getPositions()[0], m.getPositionXns()[0], m.getOctave());
			featureMap.insert(std::make_pair(&m.getFeature(), newFeature));
		}

		//Replace features in refined match
		std::vector<FeatureMatch> newRefinedMatches;
		std::vector<bool> newRefinedMatchesInliers;
		for (int i = 0, end = mRefinedMatches.size(); i!=end; ++i)
		{
			auto &match = mRefinedMatches[i];
			auto &m = match.measurement;

			//Check if feature is in new region
			auto itFeatureMap = featureMap.find(&m.getFeature());
			if (itFeatureMap != featureMap.end())
			{
				//Feature in new region, copy
				newRefinedMatches.push_back(FeatureMatch(FeatureProjectionInfo(), NULL, SlamFeatureMeasurement(itFeatureMap->second, &m.getKeyFrame(), m.getPositions(), m.getPositionXns(), m.getOctave()), match.trackLength));
				newRefinedMatchesInliers.push_back(mRefinedMatchesInliers[i]);
			}
		}
		mRefinedMatches = std::move(newRefinedMatches);

		//Add reference frame for new region
		mRegion->addKeyFrame(std::move(referenceFrameNew));

		//Update active region
		mSlam->setActiveRegion(mRegion);


		//Transform essential pose to new reference frame (where referenceFrame is at the origin)
		Pose3D &oldEssentialPose = mData->frame->getPose();
		mData->frame->setPose(std::unique_ptr<Pose3D>(new FullPose3D(FullPose3D::MakeRelativePose(referenceFrameOld->getPose(), oldEssentialPose))));

		mData->essentialReferenceFrame = mRegion->getKeyFrames().front().get();
		int newCount = checkRefinedMatchesForTriangulations();
		DTSLAM_LOG << "After checking again " << newCount << "/" << mRefinedMatches.size() << " matches ready for triangulation.\n";
	}

	/////////////////////////////////////////////
	//Add measurements to matched features
	int triangulateCount=0;
	for (int i = 0, end = mRefinedMatches.size(); i != end; ++i)
	{
		FeatureMatch &match = mRefinedMatches[i];
		SlamFeature &feature = match.measurement.getFeature();

		if (feature.getStatus() == SlamFeatureStatus::Invalid)
			continue; //Skip features that were deleted in the process

		//Add measurements
		std::unique_ptr<SlamFeatureMeasurement> measurement(new SlamFeatureMeasurement(match.measurement));

		mData->frame->getMeasurements().push_back(measurement.get());
		feature.getMeasurements().push_back(std::move(measurement));

		//Triangulate
		if (mRefinedMatchesReadyForTriangulation[i] && !feature.is3D())
		{
			SlamFeatureMeasurement *m1;
			SlamFeatureMeasurement *m2;
			float angle;
			feature.getMeasurementsForTriangulation(m1,m2,angle);
			if(m1 && m2) // && angle > mMinTriangulationAngle
			{
				mRegion->convertTo3D(feature,*m1,*m2);
				triangulateCount++;
			}
		}
	}
	DTSLAM_LOG << "Features triangulated: " << triangulateCount << "\n";

	/////////////////////////////////////////////
	//Add new 2D features
	int newKeyPointCount=0;
	cv::Size2i imageSize = mData->frame->getImage(0).size();
	for (int octave = 0; octave<mData->frame->getPyramid().getOctaveCount(); ++octave)
	{
		const int scale = 1<<octave;

		//FeatureGridIndexer<KeyPointData> keypoints = mData->frame->getKeyPoints(octave).applyNonMaximaSuppresion(scale*PatchWarper::kPatchRightSize);
		//auto &keypoints = mData->frame->getKeyPoints(octave);
		cv::Size2i tileSize(scale*FLAGS_FrameKeypointGridSize, scale*FLAGS_FrameKeypointGridSize);
		auto keypoints = FeatureGridIndexer<KeyPointData>::ApplyNonMaximaSuppresion(mData->frame->getKeyPoints(octave), imageSize, tileSize, scale*PatchWarper::kPatchRightSize);
		for (auto &keyPoint : keypoints)
		{
			int cellX = keyPoint.position.x / mFeatureCoverageMaskScale;
			int cellY = keyPoint.position.y / mFeatureCoverageMaskScale;

			if(mFeatureCoverageMask(cellY,cellX) != ECellCoveredByOld)
			{
				//Add new 2D feature
				mRegion->createFeature2D(*mData->frame, keyPoint.position, keyPoint.xn, octave);
				newKeyPointCount++;
			}
		}
	}
	DTSLAM_LOG << "New 2D features added: " << newKeyPointCount << "\n";

	/////////////////////////////////////////////
	//Add key frame
	SlamKeyFrame *res = mData->frame.get();

	mData->frame->mOriginalRegionID = mRegion->getId();
	mRegion->addKeyFrame(std::move(mData->frame));

	mStatus = ESlamMapExpanderStatus::Inactive;
	return res;
}

} /* namespace dtslam */
