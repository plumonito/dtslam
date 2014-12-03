/*
 * TwoFrameWindow.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "TwoFrameWindow.h"

#include "dtslam/SlamSystem.h"
#include "dtslam/SlamMap.h"
#include "dtslam/SlamKeyFrame.h"
#include "../shaders/DTSlamShaders.h"
#include "dtslam/EssentialEstimation.h"
#include "WindowUtils.h"

namespace dtslam {

bool TwoFrameWindow::init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize)
{
	BaseWindow::init(app, slam, imageSize);

	mKeyBindings.addBinding(false, 'q', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TwoFrameWindow::nextFrameA), "Select next frame A.");
	mKeyBindings.addBinding(false, 'a', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TwoFrameWindow::prevFrameA), "Select previous frame A.");
	mKeyBindings.addBinding(false, 'w', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TwoFrameWindow::nextFrameB), "Select next frame B.");
	mKeyBindings.addBinding(false, 's', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TwoFrameWindow::prevFrameB), "Select previous frame B.");

	mFrameATexture.create(GL_RGB, imageSize);
	mFrameBTexture.create(GL_RGB, imageSize);

	resize();
	return true;
}

void TwoFrameWindow::resize()
{
    mTiler.configDevice(cv::Rect2i(cv::Point2i(0,0),UserInterfaceInfo::Instance().getScreenSize()),2);
	mTiler.fillTiles();
	mTiler.setImageMVP(0, mImageSize);
	mTiler.setImageMVP(1, mImageSize);
}

void TwoFrameWindow::updateState()
{
	shared_lock<shared_mutex> lockRead(mSlam->getMap().getMutex());

	mValidFrameA = false;
	mValidFrameB = false;
	mDisplayText.str("");

	const SlamKeyFrame *frameA = NULL;
	const SlamKeyFrame *frameB=NULL;

	int ridx=0;
	int fidx=0;
	for(auto &region : mSlam->getMap().getRegions())
	{
		for(auto &frame : region->getKeyFrames())
		{
			if(fidx==mFrameAIdx)
			{
				mValidFrameA = true;
				frameA = frame.get();

				mDisplayText << "Left: region " << ridx << ", frame " << fidx << ", time " << frameA->getTimestamp() << "\n";
			}
			if(fidx==mFrameBIdx)
			{
				mValidFrameB = true;
				frameB = frame.get();

				mDisplayText << "Right: region " << ridx << ", frame " << fidx << ", time " << frameB->getTimestamp() << "\n";
			}

			++fidx;
		}
		++ridx;
	}

	//Handle special case when viewing the tracker frame
	if(mFrameBIdx == -1)
	{
		const SlamKeyFrame *frame = mSlam->getTracker().getFrame();
		if(frame)
		{
			mValidFrameB = true;
			frameB = frame;

			mDisplayText << "Right: tracker frame, time " << frameB->getTimestamp() << "\n";;
		}
	}

	//Update textures
	if(mValidFrameA)
		mFrameATexture.update(frameA->getColorImage());
	if(mValidFrameB)
		mFrameBTexture.update(frameB->getColorImage());

	if(mValidFrameA && mValidFrameB)
		updateState(*frameA, *frameB);
}

void TwoFrameWindow::draw()
{
	//Frame A
	mTiler.setActiveTile(0);
	mShaders->getTexture().setMVPMatrix(mTiler.getMVP());
	mShaders->getColor().setMVPMatrix(mTiler.getMVP());
	if(mValidFrameA)
	{
		mShaders->getTexture().renderTexture(mFrameATexture.getTarget(), mFrameATexture.getId(), mFrameATexture.getSize());
	}

	//Frame B
	mTiler.setActiveTile(1);
	mShaders->getTexture().setMVPMatrix(mTiler.getMVP());
	mShaders->getColor().setMVPMatrix(mTiler.getMVP());
	if(mValidFrameB)
	{
		mShaders->getTexture().renderTexture(mFrameBTexture.getTarget(), mFrameBTexture.getId(), mFrameBTexture.getSize());
	}

	//Text
	mTiler.setFullScreen();
	mShaders->getText().setMVPMatrix(mTiler.getMVP());
	mShaders->getText().setActiveFontSmall();
	mShaders->getText().setRenderCharHeight(10);
	mShaders->getText().setCaret(cv::Point2f(300,0));
	mShaders->getText().setColor(StaticColors::Green());

	{
		TextRendererStream ts(mShaders->getText());

		ts << mDisplayText.str();
	}
}

void TwoFrameWindow::nextFrameA()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mFrameAIdx++;
	if(mFrameAIdx>=mSlam->getMap().getTotalFrameCount())
		mFrameAIdx = 0;

	updateState();
}
void TwoFrameWindow::prevFrameA()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mFrameAIdx--;
	if(mFrameAIdx<0)
		mFrameAIdx = mSlam->getMap().getTotalFrameCount()-1;

	updateState();
}
void TwoFrameWindow::nextFrameB()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mFrameBIdx++;
	if(mFrameBIdx>=mSlam->getMap().getTotalFrameCount())
		mFrameBIdx = -1;

	updateState();
}
void TwoFrameWindow::prevFrameB()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mFrameBIdx--;
	if(mFrameBIdx<-1)
		mFrameBIdx = mSlam->getMap().getTotalFrameCount()-1;
	updateState();
}

} /* namespace dtslam */
