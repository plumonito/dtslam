/*
 * PnpEstimation.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef PNPESTIMATION_H_
#define PNPESTIMATION_H_

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

class PoseReprojectionError3D;
class PoseReprojectionError2D;
class EpipolarSegmentErrorForPose;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PnPRansac
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct PnPIterationData
{
	std::vector<MatchReprojectionErrors> reprojectionErrors;
};

class PnPRansac: public BaseRansac<FullPose3D, PnPIterationData, 4>
{
public:
	PnPRansac();
	~PnPRansac();

	void setData(const std::vector<FeatureMatch> *matches, const CameraModel *camera);
	int getSubconstraintCount(const int constraintIdx) const {return mMatches3D[constraintIdx]->measurement.getPositionCount();}

	std::vector<FullPose3D> modelFromMinimalSet(const std::vector<int> &constraintIndices, const std::vector<int> &constraintSubindices);
	void getInliers(const FullPose3D &model, int &inlierCount, float &errorSumSq, PnPIterationData &data);

protected:
	int mMatchCount;

	std::vector<const FeatureMatch*> mMatches3D;
	std::vector<int> mIndexes3D; //Same ordering as mMatches3D. Contains the index of the match in the original (mixed 2D and 3D) matches vector.
	std::vector<std::vector<cv::Point2f>> mImageXnNormalized3D;

	std::vector<std::unique_ptr<PoseReprojectionError3D>> mErrorFunctors3D;

	std::vector<const FeatureMatch*> mMatches2D;
	std::vector<int> mIndexes2D; //Same ordering as mMatches2D. Contains the index of the match in the original (mixed 2D and 3D) matches vector.
	std::vector<cv::Point2f> mRefXnNormalized2D;
	std::vector<std::vector<cv::Point2f>> mImageXnNormalized2D;

	std::vector<std::unique_ptr<EpipolarSegmentErrorForPose>> mErrorFunctors2D;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PnPRefiner
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class PnPRefiner
{
public:
	PnPRefiner():
		mCamera(NULL)
	{
	}

	void setCamera(const CameraModel *camera) {mCamera = camera;}
	void setOutlierThreshold(float pixelThreshold)
	{
		mOutlierPixelThreshold = pixelThreshold;
		mOutlierPixelThresholdSq = mOutlierPixelThreshold*mOutlierPixelThreshold;
	}


	bool getReprojectionErrors2D(const FeatureMatch &match,
			const cv::Matx33f &R,
			const cv::Vec3f &translation,
			MatchReprojectionErrors &errors);
	bool getReprojectionErrors3D(const FeatureMatch &match,
			const cv::Matx33f &R,
			const cv::Vec3f &translation,
			MatchReprojectionErrors &errors);

	void getInliers(const std::vector<FeatureMatch> &matches,
			const cv::Matx33f &R,
			const cv::Vec3f &translation,
			int &inlierCount,
			std::vector<MatchReprojectionErrors> &errors);

	void refinePose(const std::vector<FeatureMatch> &matches,
			cv::Matx33f &rotation,
			cv::Vec3f &translation,
			int &inlierCount,
			std::vector<MatchReprojectionErrors> &errors);

protected:
	const CameraModel *mCamera;
	float mOutlierPixelThreshold;
	float mOutlierPixelThresholdSq;
};

} /* namespace dtslam */

#endif /* PNPESTIMATION_H_ */
