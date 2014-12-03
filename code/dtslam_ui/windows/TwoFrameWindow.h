/*
 * TwoFrameWindow.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef TWOFRAMEWINDOW_H_
#define TWOFRAMEWINDOW_H_

#include "BaseWindow.h"
#include "dtslam/CameraModel.h"
#include "dtslam/Pose3D.h"

namespace dtslam
{

class SlamMap;

class TwoFrameWindow: public BaseWindow
{
public:
	TwoFrameWindow(const std::string &name):
		BaseWindow(name),
		mFrameAIdx(0), mFrameBIdx(-1), mValidFrameA(false), mValidFrameB(false)
	{}

	bool init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize);

	void resize();
	void updateState();
	void draw();

protected:
	int mFrameAIdx;
	int mFrameBIdx; //-1 means the tracker

	bool mValidFrameA;
	bool mValidFrameB;

	ViewportTiler mTiler;
	TextureHelper mFrameATexture;
	TextureHelper mFrameBTexture;

	std::stringstream mDisplayText;

	virtual void updateState(const SlamKeyFrame &frameA, const SlamKeyFrame &frameB) = 0;

	void nextFrameA();
	void prevFrameA();
	void nextFrameB();
	void prevFrameB();
};

} /* namespace dtslam */

#endif /* TWOFRAMEWINDOW_H_ */
