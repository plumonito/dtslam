/*
 * FrameLinker.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef FRAMELINKER_H_
#define FRAMELINKER_H_

#include <vector>
#include "FeatureMatcher.h"

namespace dtslam
{

class SlamKeyFrame;
class HomographyEstimation;
class PoseEstimator;

struct FrameLinkData
{
	SlamKeyFrame *frameA;
	SlamKeyFrame *frameB;
	std::vector<FeatureMatch> matches;
	FullPose3D pose;
};

class FrameLinker
{
public:
	FrameLinker();
	~FrameLinker();

	void init(const CameraModel *camera);
	std::unique_ptr<FrameLinkData> findLink(const SlamKeyFrame &frameA, const SlamKeyFrame &frameB);

	const cv::Matx23f &getSimilarity() const {return mSimilarity;}
	const std::vector<FeatureMatch> &getMatches() const {return mMatches;}
	const std::vector<bool> &getInliers() const {return mIsMatchInlier;}

	const FullPose3D &getPoseB() const {return mPoseB;}

protected:
	std::unique_ptr<HomographyEstimation> mHomographyEstimator;
	std::unique_ptr<FeatureMatcher> mMatcher;
	std::unique_ptr<PoseEstimator> mPoseEstimator;

	cv::Matx23f mSimilarity;
	std::vector<FeatureMatch> mProjections;
	std::vector<FeatureMatch> mMatches;
	std::vector<bool> mIsMatchInlier;
	FullPose3D mPoseB;
};

} /* namespace dtslam */

#endif /* FRAMELINKER_H_ */
