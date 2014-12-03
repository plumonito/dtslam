/*
 * PoseEstimation.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef POSEESTIMATION_H_
#define POSEESTIMATION_H_

#include <vector>
#include "Pose3D.h"
#include "FeatureMatcher.h"
#include "EssentialEstimation.h"
#include "RotationEstimation.h"
#include "PnpEstimation.h"
#include "FeatureMatcher.h"
#include "PoseEstimationCommon.h"

namespace dtslam
{

enum class EPoseEstimationType
{
	PureRotation,
	Essential,
	FullPose,
	Invalid
};


class PoseEstimator
{
public:
	void init(const CameraModel *camera, float outlierPixelThreshold);

	int getMinRansacIterations() const {return mMinRansacIterations;}
	void setMinRansacIterations(int value) {mMinRansacIterations=value;}

	int getMaxRansacIterations() const {return mMaxRansacIterations;}
	void setMaxRansacIterations(int value) {mMaxRansacIterations=value;}

	void setPreviousPose(const Pose3D &pose) {mPreviousPose.set(pose);}

	int getRefineOnlyInliers() const {return mRefineOnlyInliers;}
	void setRefineOnlyInliers(bool value)
	{
		mRefineOnlyInliers=value;
		assert(!value);
		//mEssentialRefiner->setRefineOnlyInliers(value);
	}

	FullPose3D &getPose() { return mPose; }
	const FullPose3D &getPose() const { return mPose; }

	EPoseEstimationType getPoseType() const { return mPoseType; }

	SlamKeyFrame *getEssentialReferenceFrame() const { return mEssentialReferenceFrame; }
	int getInlierCount() const { return mInlierCount; }

	const std::vector<MatchReprojectionErrors> &getReprojectionErrors() const { return mReprojectionErrors; }
	std::vector<MatchReprojectionErrors> &getReprojectionErrors() { return mReprojectionErrors; }

	const std::vector<MatchReprojectionErrors> &getPureRotationReprojectionErrors() const { return mPureRotationErrors; }
	std::vector<MatchReprojectionErrors> &getPureRotationReprojectionErrors() { return mPureRotationErrors; }

	const std::vector<bool> &getIsStable() const { return mIsStable; }
	std::vector<bool> &getIsStable() { return mIsStable; }

	void fitModels(std::vector<FeatureMatch> &matches);

	void refinePose(std::vector<FeatureMatch> &matches,
			const EPoseEstimationType poseType, FullPose3D &pose);



protected:
	/////////////////////////////////////////////////////////////////////////////////////
	// Members

	const CameraModel *mCamera;
	float mOutlierPixelThreshold;

	int mMinRansacIterations;
	int mMaxRansacIterations;

	bool mRefineOnlyInliers;

	FullPose3D mPreviousPose;

	std::unique_ptr<EssentialRefiner> mEssentialRefiner;
	std::unique_ptr<RotationRefiner> mRotationRefiner;
	std::unique_ptr<PnPRefiner> mPoseRefiner;

	//Results
	FullPose3D mPose;
	EPoseEstimationType mPoseType;

	int mInlierCount;
	std::vector<MatchReprojectionErrors> mReprojectionErrors;

	std::vector<bool> mIsStable;

	SlamKeyFrame *mEssentialReferenceFrame;

	int mPureRotationInlierCount;
	std::vector<MatchReprojectionErrors> mPureRotationErrors;

	/////////////////////////////////////////////////////////////////////////////////////
	// Methods
	SlamKeyFrame *findEssentialReferenceFrame(std::vector<FeatureMatch> &matches);
	void fitEssential(std::vector<FeatureMatch> &matches);
	void fitPnP(std::vector<FeatureMatch> &matches);
};

} /* namespace dtslam */

#endif /* POSEESTIMATION_H_ */
