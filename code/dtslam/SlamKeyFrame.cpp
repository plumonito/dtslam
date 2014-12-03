/*
 * SlamKeyFrame.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "SlamKeyFrame.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "Profiler.h"
#include "Pose3D.h"
#include "PatchWarper.h"
#include "SlamMap.h"
#include "flags.h"

namespace dtslam {

SlamKeyFrame::SlamKeyFrame():
		mKeyPoints(new std::vector<std::vector<KeyPointData>>())
{
}

SlamKeyFrame::SlamKeyFrame(const SlamKeyFrame &copyFrom):
		mCamera(copyFrom.mCamera), mTimestamp(copyFrom.mTimestamp), mSBI(copyFrom.mSBI), mSBIdx(copyFrom.mSBIdx), mSBIdy(copyFrom.mSBIdy),
		mColorImage(copyFrom.mColorImage), mPyramid(copyFrom.mPyramid), mKeyPoints(copyFrom.mKeyPoints),
		mMeasurements(copyFrom.mMeasurements)
{
	if(copyFrom.mPose)
		setPose(copyFrom.mPose->copy());
}

SlamKeyFrame::~SlamKeyFrame()
{
}

void SlamKeyFrame::init(const CameraModel *camera, const cv::Mat3b &imageColor, const cv::Mat1b &imageGray)
{
	mCamera = camera;

	mColorImage = imageColor;

	mPyramid.create(imageGray, FLAGS_PyramidMaxTopLevelWidth);

	//SBI
	mSBI = mPyramid.getTopLevel();
	while(mSBI.cols > FLAGS_SBIMaxWidth)
	{
		cv::Mat1b temp;
		cv::pyrDown(mSBI,temp);
		mSBI = temp;
	}

	//SBI derivatives
	cvutils::CalculateDerivatives(mSBI, mSBIdx, mSBIdy);

	//Extract key points
	mKeyPoints->resize(mPyramid.getOctaveCount());

	for(int octave=0; octave<mPyramid.getOctaveCount(); octave++)
	{
	    const int scale = 1<<octave;

	    //Create ROI
	    //cv::Rect roiRect(kNoFeatureBorderSize,kNoFeatureBorderSize, mPyramid[octave].cols-2*kNoFeatureBorderSize, mPyramid[octave].rows-2*kNoFeatureBorderSize);
	    //cv::Mat1b roi = mPyramid[octave](roiRect);

	    //FAST
	    std::vector<cv::KeyPoint> keypoints;
		//cv::FAST(roi, keypoints, kFASTThreshold, false);
	    cv::FAST(mPyramid[octave], keypoints, FLAGS_FASTThreshold, true);

	    int maxX = mPyramid[octave].cols - PatchWarper::kPatchSize - 1;
	    int maxY = mPyramid[octave].rows - PatchWarper::kPatchSize - 1;

		//Store features
	    getKeyPoints(octave).clear();
	    for(auto &keypoint : keypoints)
	    {
	    	if(keypoint.pt.x < PatchWarper::kPatchSize || keypoint.pt.y < PatchWarper::kPatchSize || keypoint.pt.x > maxX || keypoint.pt.y > maxY)
	    		continue;

	    	cv::Point2i pos((int)(scale*keypoint.pt.x), (int)(scale*keypoint.pt.y));

	    	cv::Point3f xn = camera->unprojectToWorldLUT(pos);

	    	getKeyPoints(octave).emplace_back(pos, (int)keypoint.response, octave, xn);
	    }
    }
}

std::unique_ptr<SlamKeyFrame> SlamKeyFrame::copyWithoutFeatures() const
{
	std::unique_ptr<SlamKeyFrame> newFrame(new SlamKeyFrame());

	newFrame->mCamera = mCamera;
	newFrame->mTimestamp = mTimestamp;
	newFrame->mSBI = mSBI;
	newFrame->mSBIdx = mSBIdx;
	newFrame->mSBIdy = mSBIdy;
	newFrame->mColorImage = mColorImage;
	newFrame->mPyramid = mPyramid;
	newFrame->mKeyPoints = mKeyPoints;
	
	newFrame->mOriginalRegionID = mOriginalRegionID;

	return newFrame;
}

void SlamKeyFrame::setRegion(SlamRegion *region)
{
	mRegion=region;
}

void SlamKeyFrame::setPose(std::unique_ptr<Pose3D> pose)
{
	assert(pose.get());

	mPose = std::move(pose);
}

void SlamKeyFrame::removeMeasurement(SlamFeatureMeasurement *m)
{
	auto it = mMeasurements.begin(), end = mMeasurements.end();
	for (; it != end; ++it)
	{
		if (*it == m)
			break; //Found!
	}

	if (it != end)
	{
		mMeasurements.erase(it);
	}
	else
	{
		DTSLAM_LOG << "Ahhh!!!! Attempted to remove a measurement that is not in the frame.\n";
	}
}


void SlamKeyFrame::serialize(Serializer &s, cv::FileStorage &fs) const
{
	fs << "camera" << s.addObject(mCamera);
	fs << "timestamp" << mTimestamp;

	fs << "image" << s.addImage("keyframe",mColorImage);
	fs << "region" << s.addObject(mRegion);

	fs << "pose" << s.addObject(mPose.get());

	fs << "measurements" << "[";
	for (auto &m : mMeasurements)
		fs << s.addObject(m);
	fs << "]";
}
void SlamKeyFrame::deserialize(Deserializer &s, const cv::FileNode &node)
{
	CameraModel *camera = s.getObject<CameraModel>(node["camera"]);
	cv::Mat3b imgColor = s.getImage(node["image"]);
	cv::Mat1b imgGray;
	cv::cvtColor(imgColor, imgGray, cv::COLOR_RGB2GRAY);

	init(camera, imgColor, imgGray);


	node["timestamp"] >> mTimestamp;
	mRegion = s.getObject<SlamRegion>(node["region"]);
	mPose = s.getObjectForOwner<Pose3D>(node["pose"]);

	for (const auto &n : node["measurements"])
	{
		mMeasurements.push_back(s.getObject<SlamFeatureMeasurement>(n));
	}
}

} /* namespace dtslam */
