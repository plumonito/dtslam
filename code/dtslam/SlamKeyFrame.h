/*
 * SlamKeyFrame.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef SLAMKEYFRAME_H_
#define SLAMKEYFRAME_H_

#include <gflags/gflags.h>
#include "ImagePyramid.h"
#include "CameraModel.h"
#include "Serializer.h"

namespace dtslam {

class Pose3D;
class SlamRegion;
class SlamFeatureMeasurement;

///////////////////////////////////
// Classes
struct KeyPointData
{
public:
	KeyPointData(const cv::Point2i &position_, int score_, int octave_, const cv::Point3f &xn_):
		position(position_),score(score_),octave(octave_), xn(xn_)
	{}

	cv::Point2i position;
	int score;
	int octave;
	cv::Point3f xn;
	
	//For feature indexer
	const cv::Point2i &getPosition() const { return position; }
	int getScore() const { return score; }
};

class SlamKeyFrame: public ISerializable
{
public:
	SlamKeyFrame();
	SlamKeyFrame(const SlamKeyFrame &copyFrom);
	~SlamKeyFrame();

	void init(const CameraModel *camera, const cv::Mat3b &imageColor, const cv::Mat1b &imageGray);
	std::unique_ptr<SlamKeyFrame> copyWithoutFeatures() const;

	const CameraModel &getCameraModel() const {return *mCamera;}
	void setCameraModel(const CameraModel *camera) {mCamera=camera;}

	double getTimestamp() const {return mTimestamp;}
	void setTimestamp(double value) {mTimestamp=value;}

	const cv::Mat1b &getSBI() const {return mSBI;}
	const cv::Mat1s &getSBIdx() const {return mSBIdx;}
	const cv::Mat1s &getSBIdy() const {return mSBIdy;}
	const cv::Mat3b &getColorImage() const {return mColorImage;}
	const ImagePyramid1b &getPyramid() const {return mPyramid;}
	const cv::Mat1b &getImage(int octave) const {return mPyramid[octave];}
	int getOctaveCount() const { return mPyramid.getOctaveCount(); }

	//Note: mKeyPoints never changes and is therefore a shared_ptr between all the copies of this frame (just like the images)
	//		That way pointers to key points will remain valid for all copies of the frame.
	std::vector<KeyPointData> &getKeyPoints(int octave) const {return (*mKeyPoints)[octave];}

	SlamRegion *getRegion() const {return mRegion;}
	void setRegion(SlamRegion *region);

	const Pose3D &getPose() const {return *mPose;}
	Pose3D &getPose() {return *mPose;}
	void setPose(std::unique_ptr<Pose3D> pose);

	std::vector<SlamFeatureMeasurement *> &getMeasurements() {return mMeasurements;}
	const std::vector<SlamFeatureMeasurement *> &getMeasurements() const {return mMeasurements;}

	void removeMeasurement(SlamFeatureMeasurement *m);

	/////////////////////////////////////
	// Serialization
	static const std::string GetTypeName() { return "SlamKeyFrame"; }
	const std::string getTypeName() const { return GetTypeName(); }
	void serialize(Serializer &s, cv::FileStorage &fs) const;
	void deserialize(Deserializer &s, const cv::FileNode &node);

	///////////////
	//Debug
	int mOriginalRegionID;
	///////////////

protected:
	const CameraModel * mCamera;

	double mTimestamp;

	cv::Mat1b mSBI;
	cv::Mat1s mSBIdx;
	cv::Mat1s mSBIdy;

	cv::Mat3b mColorImage;

	ImagePyramid1b mPyramid;

	//Note: mKeyPoints never changes and is therefore a shared_ptr between all the copies of this frame (just like the images)
	//		That way pointers to key points will remain valid for all copies of the frame.
	std::shared_ptr<std::vector<std::vector<KeyPointData>>> mKeyPoints;

	SlamRegion *mRegion;

	std::unique_ptr<Pose3D> mPose;

	std::vector<SlamFeatureMeasurement *> mMeasurements;
};

} /* namespace dtslam */

#endif /* SLAMKEYFRAME_H_ */
