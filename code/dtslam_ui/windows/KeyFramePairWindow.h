/*
 * KeyFramePairWindow.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef KEYFRAMEPAIRWINDOW_H_
#define KEYFRAMEPAIRWINDOW_H_

#include "TwoFrameWindow.h"
#include "dtslam/CameraModel.h"
#include "dtslam/Pose3D.h"

namespace dtslam
{

class SlamRegion;
class SlamKeyFrame;
class SlamFeature;
class SlamFeatureMeasurement;

class KeyFramePairWindow: public TwoFrameWindow
{
public:
	KeyFramePairWindow():
		TwoFrameWindow("KeyFramePairWindow"),
		mOctaveCount(0), mSelectedFeature(NULL), mSelectedMeasurementIdxA(-1), mValidPatchA(false), mValidPatchB(false), mActiveOctave(-1)
	{}

	bool init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize);
	void showHelp() const;

    void touchDown(int id, int x, int y);

	void updateState(const SlamKeyFrame &frameA, const SlamKeyFrame &frameB);
	void draw();

protected:
	int mOctaveCount;

	CameraModel mCameraA;
	FullPose3D mPoseA;

	CameraModel mCameraB;
	FullPose3D mPoseB;

	SlamFeature *mSelectedFeature;
	int mSelectedMeasurementIdxA;
	std::vector<SlamFeatureMeasurement> mMeasurementsA;
	std::vector<std::vector<cv::Point2f>> mEpiLinesA;

	bool mValidPatchA;
	TextureHelper mPatchATex;

	std::vector<SlamFeatureMeasurement> mMeasurementsB;
	std::vector<std::vector<cv::Point2f>> mEpiLinesB;

	bool mValidSelectedProjection;
	cv::Point2f mSelectedProjectionA;
	cv::Point2f mSelectedProjectionB;

	bool mValidPatchB;
	TextureHelper mPatchBTex;

	std::vector<cv::Point2f> mClickPointsA;
	std::vector<cv::Point2f> mClickPointsB;

	int mActiveOctave;
	bool mShowKeyPoints;

	void drawMeasurements(std::vector<SlamFeatureMeasurement> &measurements);

	void nextMeasurement();
	void prevMeasurement();
	void allMeasurements();
	void updateSelectedMeasurement();
	void nextOctave();
	void toggleShowKeyPoints();
};

} /* namespace dtslam */

#endif /* KEYFRAMEPAIRWINDOW_H_ */
