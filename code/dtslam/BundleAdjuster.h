/*
 * BundleAdjuster.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef BUNDLEADJUSTER_H_
#define BUNDLEADJUSTER_H_

#include <vector>
#include <unordered_set>
#include <memory>
#include <opencv2/core.hpp>
#include "SlamMap.h"

namespace dtslam {

class PoseTracker;

class BundleAdjuster
{
public:
	BundleAdjuster():
		mUseLocks(true), mIsExpanderBA(false), mRegion(NULL), mTracker(NULL)
	{}

	void setOutlierThreshold(float pixelThreshold)
	{
		mOutlierPixelThreshold = pixelThreshold;
		mOutlierPixelThresholdSq = pixelThreshold*pixelThreshold;
	}

	bool getUseLocks() const {return mUseLocks;}
	void setUseLocks(bool value) {mUseLocks = value;}

	void setIsExpanderBA(bool value) { mIsExpanderBA = value; }

	const std::unordered_set<SlamKeyFrame *> getFramesToAdjust() const {return mFramesToAdjust;}
	const std::unordered_set<SlamFeature *> getFeaturesToAdjust() const {return mFeaturesToAdjust;}

	void setRegion(SlamMap *map, SlamRegion *region) {mMap=map; mRegion=region;}
	void setTracker(PoseTracker *tracker) { mTracker = tracker; }

	void addFrameToAdjust(SlamKeyFrame &newFrame);

	bool isInlier3D(const SlamFeatureMeasurement &measurement, const std::vector<double> &pose, const cv::Vec3d &position);
	bool isInlier2D(const std::pair<SlamFeatureMeasurement,SlamFeatureMeasurement> &measurements, const std::vector<double> &poseFirst, const std::vector<double> &poseSecond);

	void getInliers(const std::unordered_map<SlamKeyFrame *, std::vector<double>> &paramsPoses,
			const std::unordered_map<SlamFeature *, cv::Vec3d> &paramsFeatures3D,
			const std::vector<SlamFeatureMeasurement> &measurements3D,
			const std::vector<std::pair<SlamFeatureMeasurement,SlamFeatureMeasurement>> &measurements2D,
			int &inlierCount);

	bool bundleAdjust();

protected:
	float mOutlierPixelThreshold;
	float mOutlierPixelThresholdSq;
	bool mUseLocks;
	bool mIsExpanderBA;

	SlamMap *mMap;
	SlamRegion *mRegion;
	PoseTracker *mTracker; //resync() will be called on this tracker if the BA is succesful
	std::unordered_set<SlamKeyFrame *> mFramesToAdjust;
	std::unordered_set<SlamFeature *> mFeaturesToAdjust;

	typedef std::pair<std::unordered_map<SlamKeyFrame *, std::vector<double>>::iterator,bool> TGetPoseParamsResult;
	TGetPoseParamsResult getPoseParams(SlamKeyFrame &frame, std::unordered_map<SlamKeyFrame *, std::vector<double>> &paramsPoses);
};

} /* namespace dtslam */

#endif /* BUNDLEADJUSTER_H_ */
