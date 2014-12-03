/*
 * SlamMapExpander.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef SLAMMAPEXPANDER_H_
#define SLAMMAPEXPANDER_H_

#include <memory>
#include <vector>
#include <unordered_set>
#include <opencv2/core.hpp>
#include "FeatureMatcher.h"
#include "PoseEstimation.h"

namespace dtslam
{

class SlamKeyFrame;
class SlamFeature;
class SlamRegion;
struct KeyPointData;
class SlamSystem;

// List of statuses to show the current stage through the UI
enum class ESlamMapExpanderStatus
{
	Inactive,
	CheckingFrame,
	AddingFrame,
	SingleFrameBA
};

class SlamMapExpander
{
public:
	//This is all the information that the expander needs from the tracker/system
	struct CheckData
	{
		bool forceAdd;
		EPoseEstimationType poseType;
		SlamKeyFrame *essentialReferenceFrame;
		std::unique_ptr<SlamKeyFrame> frame;
		std::vector<FeatureMatch> trackedFeatures;
	};

	enum ECellCoverageType
	{
		ECellEmpty=0,
		ECellCoveredByOld=127,
		ECellCoveredByNew=255
	};

	//////////////////////////////////////
	// Member functions
	bool init(const CameraModel *camera, SlamSystem *slam);
	
	//The minimum angle (in radians) before a triangulation can occur (initialized from flags)
	float getMinTriangulationAngle() const {return mMinTriangulationAngle;}

	//Returns a status that can be shown to the user to know in which stage the expander is in
	ESlamMapExpanderStatus getStatus() const { return mStatus; }
	void setStatus(ESlamMapExpanderStatus value) { mStatus = value; }

	void setRegion(SlamRegion *region) {mRegion = region;}
	SlamRegion *getRegion() const {return mRegion;}

	//This method decides whether a frame should be added as a new keyframe. Returns true if addKeyFrame should be called.
	//The thread should already have a read-lock on the map.
	bool checkFrame(std::unique_ptr<CheckData> data);

	//Adds the frame supplied to checkFrame() as a new keyframe to the region.
	//The thread should already have a write-lock on the map.
	SlamKeyFrame *addKeyFrame();

	//Used only for visualization UI
	const CheckData *getData() const {return mData.get();}
	const float getTrackerMatchAngle(int matchIdx) const {return mTrackerMatchAngles[matchIdx];}
	const cv::Mat1b &getFeatureCoverageMask() const {return mFeatureCoverageMask;}

protected:
	static const int kFeatureCoverageCells = 64;
	static const int kFeatureCoverageSize = 4;

	ESlamMapExpanderStatus mStatus;

	float mMinTriangulationAngle; //In radians

	SlamSystem *mSlam;
	SlamRegion *mRegion;
	const CameraModel *mCamera;

	std::unique_ptr<FeatureMatcher> mMatcher;
	std::unique_ptr<PoseEstimator> mPoseEstimator;

	std::unique_ptr<CheckData> mData;

	std::vector<std::vector<FeatureProjectionInfo>> mFeaturesInView;

	int mTrackerMatchesReadyForTriangulationCount;
	std::vector<float> mTrackerMatchAngles;

	std::vector<FeatureMatch> mRefinedMatches;
	std::vector<bool> mRefinedMatchesInliers; //Same indexing as mRefinedMatches
	std::vector<bool> mRefinedMatchesReadyForTriangulation; //Same indexing as mRefinedMatches
	int mReadyForTriangulationCount;

	int mFeatureCoverageMaskScale;
	cv::Mat1b mFeatureCoverageMask; //This is a grid indicating which cells have old/new/no features in the checked frame. Used to decide feature coverage.

	bool checkTrackedInfo();
	bool checkCompleteFrame();

	void prepareFeaturesInView();
	void markTrackerMatches();
	
	//This method does most of the heavy work for the expander. It searches for all possible matches between the map and the checked frame. It is slow. This limits the frame-rate of the expander.
	void findExtraMatches();

	int checkRefinedMatchesForTriangulations();

	//When starting a new region the tracker might decide that an essential model is best, but after findExtraMatches() a rotation model might be best. This changes the model.
	void changeEssentialToRotation();

	float getMatchTriangulationAngle(const FeatureMatch &match);
	float getMatchTriangulationAngleForEssential(const FeatureMatch &match, SlamKeyFrame *essentialReferenceFrame);

	void markPointAsCovered(const cv::Point2i &p, const int octave);
	bool checkFeatureCoverageRatio();
};

} /* namespace dtslam */

#endif /* SLAMMAPEXPANDER_H_ */
