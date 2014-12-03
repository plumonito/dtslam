/*
 * BaseWindow.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "BaseWindow.h"
#include <GL/freeglut.h>
#include <dtslam/log.h>
#include "../SlamDriver.h"

namespace dtslam
{

bool BaseWindow::init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize)
{
	assert(app);
	assert(slam);
	assert(imageSize.width>0 && imageSize.height>0);

	mIsInitialized = true;
	mApp = app;
	mShaders = &app->getShaders();
	mSlam = slam;
	mImageSize = imageSize;

	mKeyBindings.clear();

	return true;
}

void BaseWindow::showHelp() const
{
	DTSLAM_LOG << "\n--" << mName << " help--\n";
	mKeyBindings.showHelp();
}

void BaseWindow::keyDown(bool isSpecial, unsigned char key)
{
	mKeyBindings.dispatchKeyDown(isSpecial, key);
}

void BaseWindow::keyUp(bool isSpecial, unsigned char key)
{
	mKeyBindings.dispatchKeyUp(isSpecial, key);
}

} /* namespace dtslam */
