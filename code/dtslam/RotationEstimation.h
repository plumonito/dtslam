/*
 * RotationEstimator.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef ROTATIONESTIMATOR_H_
#define ROTATIONESTIMATOR_H_

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
// Rotation3DRansac
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Rotation3DRansac: public BaseRansac<cv::Matx33f, std::vector<int>, 2>
{
public:
	void setData(const std::vector<FeatureMatch> *matches);
	int getSubconstraintCount(const int constraintIdx) const {return mMatches->at(constraintIdx).measurement.getPositionCount();}

	std::vector<cv::Matx33f> modelFromMinimalSet(const std::vector<int> &constraintIndices, const std::vector<int> &constraintSubindices);
	//cv::Matx33f modelFromInliers(const std::vector<int> &constraintIndices);
	void getInliers(const cv::Matx33f &model, int &inlierCount, float &errorSumSq, std::vector<int> &inliers);

protected:
	const std::vector<FeatureMatch> *mMatches;
	std::vector<cv::Point3f> mFeatureDirections; //refRotation.t() * refXn
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// RotationRefiner
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class RotationRefiner
{
public:
	RotationRefiner():
		mCamera(NULL)
	{
	}

	void setCamera(const CameraModel *camera) {mCamera = camera;}
	void setOutlierThreshold(float pixelThreshold)
	{
		mOutlierPixelThreshold = pixelThreshold;
		mOutlierPixelThresholdSq = pixelThreshold*pixelThreshold;
	}

	bool getReprojectionErrors(const FeatureMatch &match,
			const cv::Vec3d &rparams,
			MatchReprojectionErrors &errors);

	void getInliers(const std::vector<FeatureMatch> &matches,
			const cv::Vec3d &rparams,
			int &inlierCount,
			std::vector<MatchReprojectionErrors> &errors);

	void refineRotation(const std::vector<FeatureMatch> &matches,
			const cv::Vec3f &center,
			cv::Matx33f &rotation,
			int &inlierCount,
			std::vector<MatchReprojectionErrors> &errors);

protected:
	const CameraModel *mCamera;
	float mOutlierPixelThreshold;
	float mOutlierPixelThresholdSq;
};

} /* namespace dtslam */

#endif /* ROTATIONESTIMATOR_H_ */
