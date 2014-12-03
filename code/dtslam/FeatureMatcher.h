/*
 * FeatureMatcher.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef FEATUREMATCHER_H_
#define FEATUREMATCHER_H_

#include <memory>
#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include "FeatureIndexer.h"
#include "SlamMap.h"
#include "PatchWarper.h"

namespace dtslam {

///////////////////////////////////
// Classes

class PatchWarper;

struct MatchCandidate
{
	cv::Point2f initialPos;
	cv::Point2f refinedPos;

	cv::Mat1b refinedPatch;
	int score;
	bool isMatch;
};

struct MatchAttempt
{
	FeatureProjectionInfo projection;
	cv::Mat1b refPatch;

	std::vector<MatchCandidate> candidates;
};

struct FeatureMatch
{
	FeatureMatch()
	{
	}

	FeatureMatch(FeatureProjectionInfo projection_, const SlamFeatureMeasurement *source, const SlamKeyFrame *frame, int octave, const std::vector<const KeyPointData *> &keyPoints, int trackLength_):
		projection(projection_), sourceMeasurement(source), measurement(&source->getFeature(), const_cast<SlamKeyFrame*>(frame), keyPoints, octave), trackLength(trackLength_)
	{}

	FeatureMatch(FeatureProjectionInfo projection_, const SlamFeatureMeasurement *source, const SlamKeyFrame *frame, int octave, const std::vector<cv::Point2f> &positions, const std::vector<cv::Point3f> &positionXns, int trackLength_) :
		projection(projection_), sourceMeasurement(source), measurement(&source->getFeature(), const_cast<SlamKeyFrame*>(frame), positions, positionXns, octave), trackLength(trackLength_)
	{}

	FeatureMatch(FeatureProjectionInfo projection_, const SlamFeatureMeasurement *source, const SlamKeyFrame *frame, int octave, const cv::Point2f &position, const cv::Point3f &positionXn, int trackLength_) :
		projection(projection_), sourceMeasurement(source), measurement(&source->getFeature(), const_cast<SlamKeyFrame*>(frame), position, positionXn, octave), trackLength(trackLength_)
	{}

	FeatureMatch(FeatureProjectionInfo projection_, const SlamFeatureMeasurement *source, SlamFeatureMeasurement m, int trackLength_) :
		projection(projection_), sourceMeasurement(source), measurement(m), trackLength(trackLength_)
	{}

	FeatureProjectionInfo projection;
	const SlamFeatureMeasurement *sourceMeasurement;
	SlamFeatureMeasurement measurement;
	int trackLength;

	void writeToMatlabLog() const;
};

class MatchingResultsData
{
public:
	friend class FeatureMatcher;

        typedef std::unordered_map<const SlamFeature *, MatchAttempt> MatchAttemptsMap; //For convenience

	MatchingResultsData():
		mAffineUsed(false), mKeyPointToProjectionAffine(cv::Matx23f::eye())
	{}

	const SlamKeyFrame &getFrame() const {return *mFrame;}
	void setFrame(const SlamKeyFrame *value) {mFrame=value;}

	const FullPose3D &getPose() const {return mPose;}
	void setPose(const Pose3D &pose) {mPose.set(pose);}

	bool affineUsed() const {return mAffineUsed;}
	const cv::Matx23f & getKeyPointToProjectionAffine() const {return mKeyPointToProjectionAffine;}

	FeatureMatch *getMatch(const SlamFeature*feature) const
	{
		auto it=mMatchedProjectionMap.find(feature);
		if(it==mMatchedProjectionMap.end())
			return NULL;
		else
			return it->second;
	}

	const std::vector<FeatureMatch> &getMatches2D() const {return mMatches2D;}
	std::vector<FeatureMatch> &getMatches2D() { return mMatches2D; }

	const std::vector<FeatureMatch> &getMatches3D() const { return mMatches3D; }
	std::vector<FeatureMatch> &getMatches3D() { return mMatches3D; }

	const MatchAttemptsMap &getMatchAttempts() const { return mMatchAttempts; }
	MatchAttemptsMap &getMatchAttempts() { return mMatchAttempts; }

	const MatchAttempt *getMatchAttempt(const SlamFeature *feature) const
	{
		auto it=mMatchAttempts.find(feature);
		if(it==mMatchAttempts.end())
			return nullptr;
		else
			return &it->second;
	}

	void createMatchMap();
	void resort2D3D();

protected:
	const SlamKeyFrame *mFrame;
	FullPose3D mPose;

	bool mAffineUsed;
	cv::Matx23f mKeyPointToProjectionAffine;

	std::unordered_map<const SlamFeature*, FeatureMatch *> mMatchedProjectionMap; //key=feature, value=pointer into either mMatches2D or mMatches3D

	std::vector<FeatureMatch> mMatches2D;
	std::vector<FeatureMatch> mMatches3D;

	MatchAttemptsMap mMatchAttempts;
};

class FeatureMatcher
{
public:
	friend class TestMatchWindow;

        typedef MatchingResultsData::MatchAttemptsMap MatchAttemptsMap;

	FeatureMatcher();

	const CameraModel &getCamera() const {return *mCamera;}
	void setCamera(const CameraModel *camera) {mCamera = camera;}

	int getSearchPixelDistance() const {return mSearchPixelDistance;}
	void setSearchDistance(const int pixels)
	{
		mSearchPixelDistance = pixels;
		mSearchPixelDistanceSq = pixels*pixels;
	}

	int getNonMaximaPixelSize() const {return mNonMaximaPixelSize;}
	void setNonMaximaPixelSize(int value) {mNonMaximaPixelSize=value;}

	int getMaxZssdScore() const {return mMaxZssdScore;}
	void setMaxZssdScore(int value) {mMaxZssdScore=value;}

	float getBestScorePercentThreshold() const {return mBestScorePercentThreshold;}
	void setBestScorePercentThreshold(float value) {mBestScorePercentThreshold=value;}

	void setFrame(const SlamKeyFrame *frame);
	void setFrame(const SlamKeyFrame *frame, const cv::Matx23f &keypointToProjection);

	void setFramePose(const Pose3D &pose);

	const SlamKeyFrame &getFrame() const {return *mFrame;}
	const FullPose3D &getPose() const {return mPose;}

	PatchWarperCache &getWarperCache() {return mWarperCache;}

	MatchAttemptsMap &getMatchAttempts() { return mMatchAttempts; }
	const MatchAttemptsMap &getMatchAttempts() const { return mMatchAttempts; }

	std::vector<FeatureMatch> &getMatches() { return mMatches; }
	const std::vector<FeatureMatch> &getMatches() const { return mMatches; }

	void clearResults();

	int findMatches(const std::vector<FeatureProjectionInfo> &projectionsToMatch);
	FeatureMatch * findMatch(const FeatureProjectionInfo &projection);

protected:
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Members

	PatchWarperCache mWarperCache;

	const CameraModel *mCamera;
	int mSearchPixelDistance;
	int mSearchPixelDistanceSq;

	struct TransformedKeypointPosition
	{
		cv::Point2f original;
		cv::Point2i transformed;
		cv::Point3f transformedXn;

		//This for the FeatureIndexer
		cv::Point2f getPosition() const { return original; }
		int getScore() const { return 0; }
	};
	//std::vector<std::vector<TransformedKeypointPosition>> mTransformedKeypoints;
	std::vector<FeatureGridIndexer<TransformedKeypointPosition>> mTransformedKeypoints;
	
	const SlamKeyFrame *mFrame;

	bool mUseAffine;
	cv::Matx23f mKeypointToProjectionAffine;

	FullPose3D mPose;

	std::vector<FeatureMatch> mMatches;

	MatchAttemptsMap mMatchAttempts;

	int mNonMaximaPixelSize;
	int mMaxZssdScore;
	float mBestScorePercentThreshold;


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Methods
	bool findMatch(const FeatureProjectionInfo &projection,
			std::vector<MatchCandidate> &candidates,
			std::vector<cv::Point2f> &positions,
			std::vector<cv::Point3f> &positionXns);

	void getPointCandidates(const SlamKeyFrame &frame,
					const int octave,
					const std::vector<cv::Point2f> &searchCenters,
					std::vector<cv::Point2f> &candidates);

	void getEpipolarCandidates(const SlamKeyFrame &frame,
					const int octave,
					const cv::Vec3f &epiPlaneNormal,
					const cv::Point3f &epipole,
					const cv::Point3f &infiniteXn,
					const cv::Point2f &infiniteUv,
					std::vector<cv::Point2f> &candidates);

	bool findMatch(const cv::Mat1b &refPatch,
					const int octave,
					const std::vector<cv::Point2f> &searchPositions,
					std::vector<MatchCandidate> &candidates,
					std::vector<cv::Point2f> &positions,
					std::vector<cv::Point3f> &positionXns);
};

} /* namespace dtslam */

#endif /* FEATUREMATCHER_H_ */
