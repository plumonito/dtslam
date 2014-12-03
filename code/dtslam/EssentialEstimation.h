/*
 * EssentialEstimation.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef ESSENTIALESTIMATION_H_
#define ESSENTIALESTIMATION_H_

#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include "BaseRansac.h"
#include "CameraModel.h"
#include "SlamMap.h"
#include "cvutils.h"
#include "FeatureMatcher.h"
#include "PoseEstimationCommon.h"

namespace dtslam {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EssentialRansac
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class EssentialReprojectionError2D;
class EpipolarSegmentErrorForPose;

struct EssentialRansacModel
{
	EssentialRansacModel(const cv::Matx33f &essential_, const Pose3D &pose_):
		essential(essential_), pose(pose_)
	{
	}

	EssentialRansacModel()
	{
	}

	cv::Matx33f essential;
	FullPose3D pose;
};

struct EssentialRansacData
{
	std::vector<MatchReprojectionErrors> reprojectionErrors;
};

class PoseReprojectionError2D;

class EssentialRansac: public BaseRansac<EssentialRansacModel, EssentialRansacData, 6>
{
public:
	EssentialRansac();
	~EssentialRansac();

	void setData(const CameraModel *camera, const std::vector<FeatureMatch> *referenceFrameMatches, const std::vector<FeatureMatch> *allMatches);
	int getSubconstraintCount(const int constraintIdx) const {return mImageXnPointsNormalized[constraintIdx].size();}

	std::vector<EssentialRansacModel> modelFromMinimalSet(const std::vector<int> &constraintIndices, const std::vector<int> &constraintSubindices);
	void getInliers(const EssentialRansacModel &model, int &inlierCount, float &errorSumSq, EssentialRansacData &data);

protected:
	const CameraModel *mCamera;
	FullPose3D mReferenceFramePose;
	const std::vector<FeatureMatch> *mReferenceFrameMatches;
	const std::vector<FeatureMatch> *mAllMatches;

	//These come from the dominant frame only
	std::vector<cv::Point2f> mRefXnPointsNormalized;
	std::vector<std::vector<cv::Point2f>> mImageXnPointsNormalized;

	//These come from all matches
	std::vector<std::unique_ptr<EpipolarSegmentErrorForPose>> mErrorFunctors;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EssentialRefiner
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class EssentialRefiner
{
public:
	EssentialRefiner():
		mCamera(NULL)
	{
	}

	void setCamera(const CameraModel *camera) {mCamera = camera;}
	void setOutlierThreshold(float pixelThreshold)
	{
		mOutlierPixelThreshold = pixelThreshold;
		mOutlierPixelThresholdSq = pixelThreshold*pixelThreshold;
	}


	//int getRefineOnlyInliers() const {return mRefineOnlyInliers;}
	//void setRefineOnlyInliers(bool value) {mRefineOnlyInliers=value;}

	bool getReprojectionErrors(const FeatureMatch &match,
			const cv::Vec3d &rparams,
			const cv::Vec3d &translation,
			MatchReprojectionErrors &errors);

	void getInliers(const std::vector<FeatureMatch> &matches,
			const cv::Vec3d &rparams,
			const cv::Vec3d &translation,
			int &inlierCount,
			std::vector<MatchReprojectionErrors> &errors);

	//rotation and translation serve as initial estimates
	void refineEssential(const std::vector<FeatureMatch> &matches,
			cv::Matx33f &rotation,
			cv::Vec3f &translation,
			int &inlierCount,
			std::vector<MatchReprojectionErrors> &errors);

protected:
	const CameraModel *mCamera;
	float mOutlierPixelThreshold;
	float mOutlierPixelThresholdSq;
	//bool mRefineOnlyInliers;
};

} /* namespace dtslam */

#endif /* POSEESTIMATION_H_ */
