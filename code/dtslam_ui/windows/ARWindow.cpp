/*
 * ARWindow.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "ARWindow.h"
#include "dtslam/SlamSystem.h"
#include "dtslam/SlamMap.h"
#include "dtslam/PoseTracker.h"
#include "../SlamDriver.h"

namespace dtslam
{
void ARWindow::showHelp() const
{
	BaseWindow::showHelp();
	DTSLAM_LOG << "Shows an overlay of the triangulated features and the cube."
		"Three display modes: show matches, show reprojected features (only the stable ones with 3 measurements), and show all reprojected features.\n";
}

bool ARWindow::init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize)
{
	BaseWindow::init(app, slam, imageSize);

	mMap = &mSlam->getMap();
	mTracker = &mSlam->getTracker();

	resize();

	mKeyBindings.addBinding(false, 't', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&ARWindow::toggleDisplayType), "Toggle display mode.");

	return true;
}

void ARWindow::updateState()
{
	shared_lock<shared_mutex> lockRead(mSlam->getMap().getMutex());

	//Clear all
	mFeatureVertices.clear();
	mFeatureColors.clear();
	mImagePoints.clear();
	mImagePointColors.clear();

	//Add features
	mTrackerCamera = &mTracker->getCamera();
	mTrackerPose = &mTracker->getCurrentPose();

	const float pointAlpha = 0.6f;

	if (mDisplayType == EDisplayType::ShowMatches)
	{
		//Get matches from tracker
		for (auto &match : mSlam->getTracker().getMatches())
		{
			bool is3D = match.measurement.getFeature().is3D();

			if (is3D)
			{
				cv::Vec4f color = StaticColors::Blue(pointAlpha);

				//Position is in 3D world coordinates
				mFeatureVertices.push_back(cvutils::PointToHomogenous(match.measurement.getFeature().getPosition()));
				mFeatureColors.push_back(color);
			}
			else
			{
				cv::Vec4f color = StaticColors::Green(pointAlpha);
				for (auto &pos : match.measurement.getPositions())
				{
					//Points are in 2D image coordinates
					mImagePoints.push_back(pos);
					mImagePointColors.push_back(color);
				}
			}
		}
	}
	else
	{
		//Determine features in view
		//Add 3D features
		for(auto &featurePtr : mTracker->getActiveRegion().getFeatures3D())
		{
			const SlamFeature &feature = *featurePtr;

			//Color
			cv::Vec4f color;
			if(feature.getMeasurements().size() > 2)
				color = StaticColors::Blue(pointAlpha);
			else
			{
				if(mDisplayType==EDisplayType::ShowStableFeatures)
					continue;
				else
					color = StaticColors::Cyan(pointAlpha);
			}

			//Add
			mFeatureVertices.push_back(cvutils::PointToHomogenous(feature.getPosition()));
			mFeatureColors.push_back(color);
		}
	}

	//Cube
	mCubeTriangleIndices.clear();
	mCubeVertices.clear();
	mCubeColors.clear();
	mCubeNormals.clear();
    if(mApp->isARCubeValid())
    {
        mApp->generateARCubeVertices(mCubeTriangleIndices, mCubeVertices, mCubeColors, mCubeNormals);
    }
}

void ARWindow::resize()
{
    mTiler.configDevice(cv::Rect2i(cv::Point2i(0,0),UserInterfaceInfo::Instance().getScreenSize()),1);
	mTiler.fillTiles();
	mTiler.setImageMVP(0, mImageSize);
}

void ARWindow::draw()
{
    mTiler.setActiveTile(0);
    mShaders->getTexture().setMVPMatrix(mTiler.getMVP());
	mShaders->getColor().setMVPMatrix(mTiler.getMVP());

	mShaders->getColorCamera().setMVPMatrix(mTiler.getMVP());
	mShaders->getColorCamera().setCamera(*mTrackerCamera);
	mShaders->getColorCamera().setPose(*mTrackerPose);

	//Draw texture
	mShaders->getTexture().renderTexture(mCurrentImageTextureTarget, mCurrentImageTextureId, mImageSize, 1.0f);

	//Draw image points
	glPointSize(8);
	mShaders->getColor().drawVertices(GL_POINTS, mImagePoints.data(), mImagePointColors.data(), mImagePoints.size());

	//Draw features
	glPointSize(8);
	mShaders->getColorCamera().drawVertices(GL_POINTS, mFeatureVertices.data(), mFeatureColors.data(), mFeatureVertices.size());

	//Draw AR cube
    if(!mCubeTriangleIndices.empty())
    {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

		mShaders->getColorCamera().drawVertices(GL_TRIANGLES, mCubeTriangleIndices.data(), mCubeTriangleIndices.size(), mCubeVertices.data(), mCubeColors.data());

	    glDisable(GL_DEPTH_TEST);
    }
}

void ARWindow::toggleDisplayType()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	switch (mDisplayType)
	{
	case EDisplayType::ShowMatches: mDisplayType = EDisplayType::ShowStableFeatures; break;
	case EDisplayType::ShowStableFeatures: mDisplayType = EDisplayType::ShowAllFeatures; break;
	case EDisplayType::ShowAllFeatures: mDisplayType = EDisplayType::ShowMatches; break;
	}
	updateState();
}

} /* namespace dtslam */
