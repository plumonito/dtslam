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

#ifndef TESTMATCHWINDOW_H_
#define TESTMATCHWINDOW_H_

#include "BaseWindow.h"
#include "dtslam/CameraModel.h"
#include "dtslam/Pose3D.h"

namespace dtslam
{

class SlamRegion;
class SlamKeyFrame;
class SlamFeature;
class SlamFeatureMeasurement;

class TestMatchWindow: public BaseWindow
{
public:
	TestMatchWindow():
		BaseWindow("TestMatchWindow"),
		mFrameAIdx(0), mFrameBIdx(-1), mUseRefiner(false), mUseEpipolar(false), mShowKeyPoints(true), mValidFrameA(false), mValidFrameB(false)
	{}

	bool init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize);
	void showHelp() const;

    void touchDown(int id, int x, int y);

	void resize();
	void updateState();
	void draw();

protected:
	ViewportTiler mTiler;

	SlamRegion *mRegion;

	int mFrameAIdx;
	int mFrameBIdx; //-1 means the tracker
	int mCandidateIdx; //-1 means the best
	bool mUseRefiner;
	bool mUseEpipolar;
	bool mShowKeyPoints;

	bool mValidFrameA;
	CameraModel mCameraA;
	FullPose3D mPoseA;
	TextureHelper mFrameATexture;

	bool mValidFrameB;
	CameraModel mCameraB;
	FullPose3D mPoseB;
	TextureHelper mFrameBTexture;

	int mOctaveCount;
	int mActiveOctave;

	cv::Point2f mClickPointA;
	cv::Point2f mClickPointB;

	bool mMatchFound;
	cv::Point2f mMatchPosA;
	cv::Point2f mMatchStartPosB;
	cv::Point2f mMatchPosB;
	float mMatchScore;
	std::vector<cv::Point2f> mCandidatePositions;
	std::vector<cv::Point2f> mKeyPointPositions;

	bool mValidPatchA;
	cv::Mat1b mPatchA;
	TextureHelper mPatchATexture;

	bool mValidPatchB;
	cv::Mat1b mPatchB;
	TextureHelper mPatchBTexture;

	std::vector<std::vector<cv::Point2f>> mEpiLinesB;

	void nextFrameA();
	void prevFrameA();
	void nextFrameB();
	void prevFrameB();
	void nextOctave();
	void nextCandidate();
	void toggleUseRefiner();
	void toggleUseEpipolar();
	void toggleShowKeyPoints();
	void logPatches();
};

} /* namespace dtslam */

#endif /* KEYFRAMEPAIRWINDOW_H_ */
