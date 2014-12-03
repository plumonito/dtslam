/*
 * PoseTracker.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef POSETRACKER_H_
#define POSETRACKER_H_

#include <memory>
#include <vector>
#include <list>
#include <unordered_map>
#if !defined(_MSC_VER)
#   include <map>
#endif

#include "PatchWarper.h"
#include "SlamMap.h"
#include "FeatureMatcher.h"

#include "HomographyEstimation.h"
#include "PoseEstimation.h"

namespace dtslam {

///////////////////////////////////
// Classes
class SlamKeyFrame;
class HomographyEstimation;

class PoseTracker
{
public:
    typedef FeatureMatcher::MatchAttemptsMap MatchAttemptsMap;

	void init(const CameraModel *camera, const cv::Size &imageSize, int octaveCount);
	void resetTracking(const Pose3D &initialPose);

	int getMatcherSearchRadius() const {return mMatcherSearchRadius;}

	const CameraModel &getCamera() const {return *mCamera;}

	SlamRegion &getActiveRegion() const {return *mActiveRegion;}
	void setActiveRegion(SlamRegion *value) {mActiveRegion = value;}

	bool trackFrame(std::unique_ptr<SlamKeyFrame> frame);

	void resync();

	bool isLost() const;

	const FullPose3D &getCurrentPose() const {return mCurrentPose;}
	void setCurrentPose(const Pose3D &pose) {mCurrentPose.set(pose);}

	EPoseEstimationType getPoseType() const { return mPoseType; }

	const cv::Size getImageSize() const {return mImageSize;}
	const int getOctaveCount() const {return mOctaveCount;}

	const SlamKeyFrame *getFrame() const { return mFrame.get(); }

	const cv::Matx23f &getFrameToLastSimilarity() const { return mSimilarityInv; }
	const cv::Matx23f &getLastToFrameSimilarity() const { return mSimilarity; }

	const MatchAttemptsMap &getMatchAttempts() const { return mMatchAttempts; }
	const FeatureMatch *getMatch(const SlamFeature *feature) const;
	const std::vector<FeatureMatch> &getMatches() const { return mMatches; }

	int getMatchInlierCount() const { return mMatchInlierCount; }
	const std::vector<MatchReprojectionErrors> &getReprojectionErrors() const { return mReprojectionErrors; }
	const std::vector<MatchReprojectionErrors> &getReprojectionErrorsForRotation() const { return mReprojectionErrorsForRotation; }

	SlamKeyFrame *getEssentialReferenceFrame() const { return mEssentialReferenceFrame; }

protected:
	/////////////////////////////////////////////////////
	// Protected members

	//bool mIsLost;

	SlamRegion *mActiveRegion;
	const CameraModel *mCamera;
	cv::Size mImageSize;
	int mOctaveCount;

	std::unique_ptr<FeatureMatcher> mMatcher;
	std::unique_ptr<HomographyEstimation> mHomographyEstimator;
	std::unique_ptr<PoseEstimator> mPoseEstimator;

	int mMatcherSearchRadius; //Contrary to the flag, this is in image pixel units

	cv::Size2i mFeatureSortGridSize;

	FullPose3D mCurrentPose;
	EPoseEstimationType mPoseType;

	//Data from the previous frame
	//Only inliers are kept here
	//std::unique_ptr<FrameTrackingData> mLastTrackedFrameDat;
	std::unique_ptr<SlamKeyFrame> mLastFrame;
	std::vector<FeatureMatch> mLastMatches;

	//Data from the current frame
	std::unique_ptr<SlamKeyFrame> mFrame;

	cv::Matx23f mSimilarity;
	cv::Matx23f mSimilarityInv;

	std::vector<std::vector<FeatureProjectionInfo>> mFeaturesInView; //Outer vector is of octaves, inner of projections

    MatchAttemptsMap mMatchAttempts;
	std::vector<FeatureMatch> mMatches;
	std::unordered_map<const SlamFeature*, FeatureMatch *> mMatchMap; //key=feature, value=pointer into either mMatches

	int mMatchInlierCount;
	std::vector<MatchReprojectionErrors> mReprojectionErrors;	//Errors for the essential or pnp model. Same index as mMatches
	std::vector<MatchReprojectionErrors> mReprojectionErrorsForRotation;	//Errors for the pure rotation model. Same index as mMatches
	SlamKeyFrame *mEssentialReferenceFrame;

	//std::unique_ptr<MatchingResultsData> mMatchingData;
	//std::unique_ptr<PoseEstimatorData> mEstimatorData;


	/////////////////////////////////////////////////////
	// Protected methods

	bool estimateSimilarityFromLastFrame(const SlamKeyFrame &frame, cv::Matx23f &similarity);

	typedef std::vector<FeatureProjectionInfo *> TSortedFeaturesCell;
	void sortFeaturesInOctave(const std::vector<FeatureProjectionInfo> &previousMatches, 
			const std::vector<FeatureProjectionInfo> &featuresInOctave,
			std::vector<TSortedFeaturesCell> &featureGrid,
			std::vector<FeatureProjectionInfo *> &features2D);
	void matchFeaturesInOctave(int maxFeatureCount,
			std::vector<TSortedFeaturesCell> &featureGrid,
			std::vector<FeatureProjectionInfo *> &features2D);

	void updateFeatureProjections(const Pose3D &pose,
			std::vector<FeatureProjectionInfo> &featureProjections);
	void findMatches(const int octave, const std::vector<std::pair<SlamFeature *, KeyPointData*>> &matches);
};

} /* namespace dtslam */

#endif /* POSETRACKER_H_ */
