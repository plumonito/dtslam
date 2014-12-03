/*
 * SlamMap.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef SLAMMAP_H_
#define SLAMMAP_H_

#include <vector>
#include <memory>
#include <atomic>
#include <opencv2/core.hpp>
#include "CameraModel.h"
#include "SlamKeyFrame.h"
#include "Serializer.h"

#include "CameraModel.h"
#include "Pose3D.h"
#include "shared_mutex.h"
namespace dtslam {

class SlamMap;
class SlamRegion;
class Slam2DSection;
class Pose3D;
class SlamKeyFrame;
class SlamFeature;
class SlamFeatureMeasurement;
class FeatureProjectionInfo;
struct EpipolarProjection;

//This is the top class that holds all data about the world
class SlamMap: public ISerializable
{
public:
	SlamMap(): mNextRegionId(0)
	{
	}

	shared_mutex &getMutex() {return mMutex;}
	std::mutex &getLongOperationMutex() {return mLongOperationMutex;}

	void clear();

	const std::vector<std::unique_ptr<SlamRegion>> &getRegions() const {return mRegions;}
	SlamRegion *createRegion();
	void mergeRegions(SlamRegion &regionA, SlamRegion &regionB, const cv::Matx33f &Rrel, float scale, const cv::Vec3f &trel);

	int getTotalFrameCount() const;
	int getTotalFeature3DCount() const;
	int getTotalFeature2DCount() const;

	/////////////////////////////////////
	// Serialization
	static const std::string GetTypeName() { return "SlamMap"; }
	const std::string getTypeName() const { return GetTypeName(); }
	void serialize(Serializer &s, cv::FileStorage &fs) const;
	void deserialize(Deserializer &s, const cv::FileNode &node);

protected:
	std::vector<std::unique_ptr<SlamRegion>> mRegions;
	int mNextRegionId;

	//This is locked by the expander before a long operation and by the bundler before the
	//write lock. It ensures that the request for a write lock will not block the pose tracker
	//because the expander has a long read lock.
	std::mutex mLongOperationMutex;

	shared_mutex mMutex;
};

//A SlamRegion is a connected set of KeyFrames with euclidean registration.
class SlamRegion : public ISerializable
{
public:
	friend class SlamMap;

	SlamRegion() : mShouldBundleAdjust(false), mFirstTriangulationFrame(NULL)
	{}

	int getId() const {return mId;}
	void setId(int id) {mId = id;}

	const std::vector<std::unique_ptr<SlamKeyFrame>> &getKeyFrames() const {return mKeyFrames;}
	const std::vector<std::unique_ptr<SlamFeature>> &getFeatures2D() const {return mFeatures2D;}
	const std::vector<std::unique_ptr<SlamFeature>> &getFeatures3D() const {return mFeatures3D;}

	SlamKeyFrame *getPreviousRegionSourceFrame() const {return mPreviousRegionSourceFrame;}
	void setPreviousRegionSourceFrame(SlamKeyFrame *frame) {mPreviousRegionSourceFrame = frame;}

	SlamKeyFrame *getFirstTriangulationFrame() const {return mFirstTriangulationFrame;}

	bool getShouldBundleAdjust() const { return mShouldBundleAdjust; }
	void setShouldBundleAdjust(bool value) { mShouldBundleAdjust = value; }

	const std::atomic<bool> &getAbortBA() const { return mAbortBA; }
	void setAbortBA(bool value) { mAbortBA = value; }

	void addKeyFrame(std::unique_ptr<SlamKeyFrame> newKeyFrame);
	void addFeature3D(std::unique_ptr<SlamFeature> newFeature);

	void getFeaturesInView(const Pose3D &pose, const CameraModel &camera, const int octaveCount, bool onlyNearest2DSection, const std::unordered_set<SlamFeature*> &featuresToIgnore, std::vector<std::vector<FeatureProjectionInfo>> &featuresInView);
	void getFeaturesInView(const SlamKeyFrame &frame, bool onlyNearest2DSection, const std::unordered_set<SlamFeature*> &featuresToIgnore, std::vector<std::vector<FeatureProjectionInfo>> &featuresInView)
	{
		getFeaturesInView(frame.getPose(), frame.getCameraModel(), frame.getPyramid().getOctaveCount(), onlyNearest2DSection, featuresToIgnore, featuresInView);
	}

	SlamFeature *createFeature2D(SlamKeyFrame &keyFrame, const cv::Point2f &position, const cv::Point3f &positionXn, int octave);
	void convertTo3D(SlamFeature &feature, SlamFeatureMeasurement &m1, SlamFeatureMeasurement &m2);

	void moveToGarbage(SlamFeature &feature);

	static FeatureProjectionInfo Project3DFeature(const Pose3D &pose, const cv::Vec3f &poseCenter, const CameraModel &camera, int octaveCount, const SlamFeature &feature);
	static FeatureProjectionInfo Project2DFeature(const Pose3D &pose, const cv::Vec3f &poseCenter, const CameraModel &camera, SlamFeatureMeasurement &measurement);

	static EpipolarProjection CreateEpipolarProjection(const Pose3D &refPose, const cv::Point3f refXn, const Pose3D &imgPose);

	/////////////////////////////////////
	// Serialization
	static const std::string GetTypeName() { return "SlamRegion"; }
	const std::string getTypeName() const { return GetTypeName(); }
	void serialize(Serializer &s, cv::FileStorage &fs) const;
	void deserialize(Deserializer &s, const cv::FileNode &node);

protected:
	int mId;

	std::vector<std::unique_ptr<SlamKeyFrame>> mKeyFrames;
	std::vector<std::unique_ptr<SlamFeature>> mFeatures2D;
	std::vector<std::unique_ptr<SlamFeature>> mFeatures3D;
	
	std::vector<std::unique_ptr<SlamFeature>> mGarbageFeatures;

	SlamKeyFrame *mPreviousRegionSourceFrame; //This frame belongs to another region and is the same as the first frame in this region. NULL for the first region.

	SlamKeyFrame *mFirstTriangulationFrame;

	std::atomic<bool> mShouldBundleAdjust;
	std::atomic<bool> mAbortBA;
};

enum class SlamFeatureStatus
{
	Invalid,
	NotTriangulated,
	TwoViewTriangulation,
	ThreeViewAgreement,
	ThreeViewDisagreement,
	MultiViewAgreement,
	MultiViewDisagreement
};

//Represents a feature (2D or 3D) that has been observed and can potentially be matched.
//In 3D it is assumed to be a small planar patch.
class SlamFeature : public ISerializable
{
public:
	friend SlamRegion;
	friend Slam2DSection;

	SlamRegion *getRegion() { return mRegion; }
	void setRegion(SlamRegion *region) { mRegion = region; }

	bool is3D() const {return mIs3D;}
	
	SlamFeatureStatus getStatus() const { return mStatus; }
	void setStatus(SlamFeatureStatus value) { mStatus = value; }
	void setStatus(int inlierMeasurementCount);

	const cv::Point3f &getPosition() const {assert(is3D()); return mPosition;}
	void setPosition(const cv::Point3f &value) {mPosition=value;}

	cv::Point3f getNormal() const {return mNormal;}
	cv::Point3f getPlusOneOffset() const {return mPlusOneOffset;}
	cv::Point3f getPositionPlusOne() const {return mPosition+mPlusOneOffset;}

	int getOctaveFor2DFeature() const;

	std::vector<std::unique_ptr<SlamFeatureMeasurement>> &getMeasurements() {return mMeasurements;}
	const std::vector<std::unique_ptr<SlamFeatureMeasurement>> &getMeasurements() const {return mMeasurements;}

	SlamFeatureMeasurement *getBestMeasurementForMatching(const cv::Vec3f &poseCenter) const;

	float static GetTriangulationAngle(const SlamFeatureMeasurement &m1, const SlamFeatureMeasurement &m2);
	float getMinTriangulationAngle(const SlamFeatureMeasurement &m1) const;
	void getMeasurementsForTriangulation(SlamFeatureMeasurement *&m1, SlamFeatureMeasurement *&m2, float &angle) const;
	void getMeasurementsForTriangulation(const SlamFeatureMeasurement &m1, SlamFeatureMeasurement *&m2, float &angle) const;

	//DEBUG:
	int mOriginalRegionId;

	/////////////////////////////////////
	// Serialization
	static const std::string GetTypeName() { return "SlamFeature"; }
	const std::string getTypeName() const { return GetTypeName(); }
	void serialize(Serializer &s, cv::FileStorage &fs) const;
	void deserialize(Deserializer &s, const cv::FileNode &node);

protected:
	SlamRegion *mRegion;

	bool mIs3D; //If the feature is 2D this points to the section it belongs to.
	SlamFeatureStatus mStatus;

	cv::Point3f mPosition;	//Position in space of the center of the patch.
							//If no 3D info is available this has unit norm and it indicates a direction.
							//When mPosition is projected to an image it provides the patch center.
							//For the first frame it is at the beginning (kPatchCenterOffset, kPatchCenterOffset).
	cv::Point3f mNormal; //Must always have unit norm.
	cv::Point3f mPlusOneOffset;	//Relative offset that represents the offset of 1 pixel in the source image.
									//It determines the feature's scale.
									//When (mPosition+mPlusOneOffset) is projected onto the source image it provides
									//the position 1 pixel to the right of the center. This is used to choose the scale where
									//to match the feature.
									//If the feature is 2D then this is not used.

	std::vector<std::unique_ptr<SlamFeatureMeasurement>> mMeasurements;
};

class SlamFeatureMeasurement: public ISerializable
{
public:
	SlamFeatureMeasurement()
	{}

	SlamFeatureMeasurement(SlamFeature *feature, SlamKeyFrame *keyFrame, const cv::Point2f &position, const cv::Point3f &positionXn, int octave):
		mFeature(feature), mKeyFrame(keyFrame), mPositions(1,position), mPositionXns(1,positionXn), mOctave(octave)
	{
	}

	SlamFeatureMeasurement(SlamFeature *feature, SlamKeyFrame *keyFrame, const std::vector<cv::Point2f> &positions, const std::vector<cv::Point3f> &positionXns, int octave):
		mFeature(feature), mKeyFrame(keyFrame), mPositions(positions), mPositionXns(positionXns), mOctave(octave)
	{
	}

	SlamFeatureMeasurement(SlamFeature *feature, SlamKeyFrame *keyFrame, const std::vector<const KeyPointData *> &keyPoints, int octave):
		mFeature(feature), mKeyFrame(keyFrame), mOctave(octave)
	{
		for(auto &kpPtr : keyPoints)
		{
			auto &kp = *kpPtr;
			mPositions.push_back(kp.position);
			mPositionXns.push_back(kp.xn);
		}
	}

	SlamFeature &getFeature() const {return *mFeature;}
	SlamKeyFrame &getKeyFrame() const {return *mKeyFrame;}
	const Pose3D &getFramePose() const {return mKeyFrame->getPose();} //Shortcut
	const CameraModel &getCamera() const {return mKeyFrame->getCameraModel();} //Shortcut

	int getPositionCount() const {return mPositions.size();}

	const std::vector<cv::Point2f> &getPositions() const {return mPositions;}
	std::vector<cv::Point2f> &getPositions() {return mPositions;}
	const std::vector<cv::Point3f> &getPositionXns() const {return mPositionXns;}
	std::vector<cv::Point3f> &getPositionXns() {return mPositionXns;}

	const cv::Point2f &getUniquePosition() const
	{
		if (mPositions.size() != 1)
		{
			assert(mPositions.size() == 1);
		}
		return mPositions[0];
	}
	const cv::Point3f &getUniquePositionXn() const 
	{
		if (mPositionXns.size() != 1)
		{
			assert(mPositionXns.size() == 1);
		}
		return mPositionXns[0];
	}

	int getOctave() const {return mOctave;}
	const cv::Mat1b &getImage() const {return mKeyFrame->getImage(mOctave);}

	/////////////////////////////////////
	// Serialization
	static const std::string GetTypeName() { return "Measurement"; }
	const std::string getTypeName() const { return GetTypeName(); }
	void serialize(Serializer &s, cv::FileStorage &fs) const;
	void deserialize(Deserializer &s, const cv::FileNode &node);

protected:
	SlamFeature *mFeature;
	SlamKeyFrame *mKeyFrame;

	std::vector<cv::Point2f> mPositions;
	std::vector<cv::Point3f> mPositionXns;
	int mOctave;
};

//Feature projections
struct PointProjection
{
	std::vector<cv::Point2f> positions;
};

struct EpipolarProjection
{
	cv::Vec3f epiPlaneNormal;
	cv::Point3f minDepthXn;
	cv::Point3f infiniteXn;
};

enum class EProjectionType
{
	Invalid,
	EpipolarLine,
	PointProjection,
	PreviousMatch
};

class FeatureProjectionInfo
{
public:
	//////////////////////////
	//Constructors
	FeatureProjectionInfo() : mType(EProjectionType::Invalid)
	{}

	static FeatureProjectionInfo CreatePoint(SlamFeature *feature, SlamFeatureMeasurement *sourceMeasurement, int octave, const cv::Point2f &point) 
	{
		return FeatureProjectionInfo(EProjectionType::PointProjection, feature, sourceMeasurement, octave, 0, { { point } });
	}

	static FeatureProjectionInfo CreatePreviousMatch(SlamFeature *feature, SlamFeatureMeasurement *sourceMeasurement, int octave, int trackLength, const std::vector<cv::Point2f> &positions)
	{
		return FeatureProjectionInfo(EProjectionType::PointProjection, feature, sourceMeasurement, octave, trackLength, { positions });
	}

	static FeatureProjectionInfo CreateEpipolar(SlamFeature *feature, SlamFeatureMeasurement *sourceMeasurement, int octave, const cv::Vec3f &epiPlaneNormal, const cv::Point3f &minDepthXn, const cv::Point3f &infiniteXn)
	{
		return FeatureProjectionInfo(feature, sourceMeasurement, octave, {epiPlaneNormal,minDepthXn,infiniteXn});
	}

	////////////////////////////////////////
	// Properties for all projection types
	EProjectionType getType() const { return mType; }

	SlamFeature &getFeature() const {return *mFeature;}
	SlamFeatureMeasurement &getSourceMeasurement() const {return *mSourceMeasurement;}

	int getOctave() const {return mOctave;}
	int getTrackLength() const {return mTrackLength;}

	////////////////////////////////////////
	// Type specific data
	const PointProjection &getPointData() const { assert(mType != EProjectionType::EpipolarLine); return mPointData; }
	const EpipolarProjection &getEpipolarData() const { assert(mType == EProjectionType::EpipolarLine); return mEpipolarData; }

protected:
	FeatureProjectionInfo(EProjectionType type, SlamFeature *feature_, SlamFeatureMeasurement *sourceMeasurement_, int octave_, int trackLength_, const PointProjection &pointData) :
		mType(type), mFeature(feature_), mSourceMeasurement(sourceMeasurement_), mOctave(octave_), mTrackLength(trackLength_), mPointData(pointData)
	{
	}

	FeatureProjectionInfo(SlamFeature *feature_, SlamFeatureMeasurement *sourceMeasurement_, int octave_, const EpipolarProjection &epipolarData) :
		mType(EProjectionType::EpipolarLine), mFeature(feature_), mSourceMeasurement(sourceMeasurement_), mOctave(octave_), mTrackLength(0), mEpipolarData(epipolarData)
	{
	}

	EProjectionType mType;

	SlamFeature *mFeature;
	SlamFeatureMeasurement *mSourceMeasurement;

	int mOctave;
	int mTrackLength;

	PointProjection mPointData;
	EpipolarProjection mEpipolarData;
};

} /* namespace dtslam */

#endif /* SLAMMAP_H_ */
