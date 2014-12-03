/*
 * FrameLinker.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "FrameLinker.h"
#include "SlamMap.h"
#include "HomographyEstimation.h"
#include "PoseEstimation.h"
#include "PoseTracker.h"
#include "flags.h"

#include <opencv2/imgproc.hpp>

namespace dtslam
{

FrameLinker::FrameLinker()
{
}

FrameLinker::~FrameLinker()
{
}

void FrameLinker::init(const CameraModel *camera)
{
	mHomographyEstimator.reset(new HomographyEstimation());

	mMatcher.reset(new FeatureMatcher());
	mMatcher->setCamera(camera);
	mMatcher->setMaxZssdScore(FLAGS_MatcherMaxZssdScore);
	mMatcher->setSearchDistance((1<<2)*8);

	mPoseEstimator.reset(new PoseEstimator());
	mPoseEstimator->init(camera, (float)FLAGS_TrackerOutlierPixelThreshold);
}

std::unique_ptr<FrameLinkData> FrameLinker::findLink(const SlamKeyFrame &frameA, const SlamKeyFrame &frameB)
{
	//Clear
	mMatches.clear();

	//Estimate similarity between the frames
	mSimilarity = cv::Matx23f::eye(); //Identity is used as the initial guess
	if(!mHomographyEstimator->estimateSimilarityDirect(frameA.getSBI(), frameA.getSBIdx(), frameA.getSBIdy(), frameB.getSBI(), mSimilarity))
		return nullptr;

	//Scale similarity matrix
	const int scale = frameB.getImage(0).cols / frameB.getSBI().cols;
	mSimilarity(0,2) *= scale;
	mSimilarity(1,2) *= scale;
	cv::Matx23f simInv;
	cv::invertAffineTransform(mSimilarity, simInv);

	//First pass
	//Find matches
	mMatcher->clearResults(); //Reset results
	mMatcher->setFrame(&frameB, simInv);
	mMatcher->setFramePose(frameA.getPose());
	for(auto m : frameA.getMeasurements())
	{
		if(!m->getFeature().is3D() || m->getPositionCount()>1)
			continue;

		FeatureProjectionInfo projection = FeatureProjectionInfo::CreatePreviousMatch(&m->getFeature(), m, m->getOctave(), 0, m->getPositions());

		FeatureMatch *match = mMatcher->findMatch(projection);
		if(match)
			mMatches.push_back(*match);

		//Debug: This to show the results of the similarity, i.e. where the features are projected
//		cv::Point2f posB = cvutils::AffinePoint(mSimilarity, m->getUniquePosition());
//		cv::Point3f xnB = frameB.getCameraModel().unprojectToWorld(posB);
//		mMatches.push_back(FeatureMatch(m, &frameB, m->getOctave(), posB, xnB, 0));
	}

	if(mMatches.size() < 10)
		return nullptr;

	//Estimate pose
	mPoseEstimator->setPreviousPose(frameA.getPose());
	mPoseEstimator->fitModels(mMatches);

	if(mPoseEstimator->getPoseType() == EPoseEstimationType::Invalid ||mPoseEstimator->getInlierCount() < 10)
		return nullptr;

	mIsMatchInlier = std::move(mPoseEstimator->getIsStable());

	//Pose after pass 1
	mPoseB = mPoseEstimator->getPose();

	//Second pass
	//Do not match inliers again
	std::unordered_set<SlamFeature*> featuresToIgnore;
	std::vector<FeatureMatch> inlierMatches;
	for(int i=0,end=mMatches.size(); i!=end; ++i)
	{
		if(mIsMatchInlier[i])
		{
			inlierMatches.push_back(mMatches[i]);
			featuresToIgnore.insert(&mMatches[i].measurement.getFeature());
		}
	}
	mMatches = inlierMatches; //Forget the outliers, maybe we'll match them again

	int octaveCount = frameB.getPyramid().getOctaveCount();
	std::vector<std::vector<FeatureProjectionInfo>> projections;
	frameA.getRegion()->getFeaturesInView(mPoseB, frameB.getCameraModel(), octaveCount, false, std::unordered_set<SlamFeature*>(), projections);

	for(int octave=octaveCount-1; octave>=0; --octave)
	{
		for(auto &projection : projections[octave])
		{
			auto &feature = projection.getFeature();
			if(!feature.is3D() || featuresToIgnore.find(&feature)!=featuresToIgnore.end())
				continue;

			FeatureMatch *match = mMatcher->findMatch(projection);
			if(match)
				mMatches.push_back(*match);
		}
	}

	//Estimate pose
	mPoseEstimator->setPreviousPose(mPoseB);
	mPoseEstimator->fitModels(mMatches);

	if (mPoseEstimator->getPoseType() == EPoseEstimationType::Invalid || mPoseEstimator->getInlierCount() < 100)
		return nullptr;

	mIsMatchInlier = std::move(mPoseEstimator->getIsStable());

	//Pose after pass 2
	mPoseB = mPoseEstimator->getPose();

	//Create link data
	std::unique_ptr<FrameLinkData> data(new FrameLinkData());
	data->frameA = const_cast<SlamKeyFrame*>(&frameA);
	data->frameB = const_cast<SlamKeyFrame*>(&frameB);
	for(int i=0,end=mMatches.size(); i!=end; ++i)
	{
		if(mIsMatchInlier[i])
			data->matches.push_back(mMatches[i]);
	}
	data->pose = mPoseB;
	return std::move(data);
}

} /* namespace dtslam */
