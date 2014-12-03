/*
 * MapExpanderWindow.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "MapExpanderWindow.h"
#include "dtslam/SlamMapExpander.h"
#include "dtslam/SlamSystem.h"
#include "../shaders/DTSlamShaders.h"

namespace dtslam
{

void MapExpanderWindow::showHelp() const
{
	BaseWindow::showHelp();
	DTSLAM_LOG << "Shows the results of the SlamMapExpander. The squares represent the tracked features (yellow=2D not ready for triangulation, green=2D ready for triangulation,blue=3D). The number for each feature is the disparity with respect to the farthest measurement."
			<< "The overlayed gray squares represent feature coverage: transparent=empty, dark gray=covered by old features, light gray=covered by features not yet in the map";
}

bool MapExpanderWindow::init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize)
{
	BaseWindow::init(app, slam, imageSize);

	mMapExpander = &mSlam->getMapExpander();
	assert(mMapExpander);

	mFeatureCoverageTexture.create(GL_RGB, mMapExpander->getFeatureCoverageMask().size());
	mFeatureCoverageImage.create(mMapExpander->getFeatureCoverageMask().size());

    glBindTexture(GL_TEXTURE_2D, mFeatureCoverageTexture.getId());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	//Add bindings
	//mKeyBindings.push_back(WindowKeyBinding(false,'*',static_cast<WindowKeyBindingFunction>(&MatchesWindow::resetDebugMatchIdx),"View all matches."));

	resize();
	return true;
}

void MapExpanderWindow::resize()
{
    mTiler.configDevice(cv::Rect2i(cv::Point2i(0,0),UserInterfaceInfo::Instance().getScreenSize()),1);
	mTiler.fillTiles();
	mTiler.setImageMVP(0, mImageSize);
}

void MapExpanderWindow::updateState()
{
	shared_lock<shared_mutex> lockRead(mSlam->getMap().getMutex());

	const float foregroundAlpha = 0.8f;

	mOctaveCount = mSlam->getTracker().getOctaveCount();

	//Update texture
	const cv::Mat1b &grayMask = mMapExpander->getFeatureCoverageMask();
	for(int v=0; v<grayMask.rows; ++v)
	{
		auto grayRow = grayMask[v];
		auto colorRow = mFeatureCoverageImage[v];
		for(int u=0; u<grayMask.cols; ++u)
		{
			switch(grayRow[u])
			{
			case SlamMapExpander::ECellEmpty: colorRow[u] = cv::Vec3b(0,0,0); break;
			case SlamMapExpander::ECellCoveredByOld: colorRow[u] = cv::Vec3b(0,0,255); break;
			case SlamMapExpander::ECellCoveredByNew: colorRow[u] = cv::Vec3b(255,255,0); break;
			}
		}
	}
	mFeatureCoverageTexture.update(mFeatureCoverageImage);

	//Copy tracked features
	mMatchesToDraw.clear();
	if(!mMapExpander->getData())
		return;
	auto &expanderData = *mMapExpander->getData();

	for(int i=0,endI=expanderData.trackedFeatures.size(); i<endI; ++i)
	{
		const auto &match = expanderData.trackedFeatures[i];
		const auto &m = match.measurement;
		const SlamFeature &feature = m.getFeature();

		for(int j=0,endJ=m.getPositions().size(); j!=endJ; ++j)
		{
			mMatchesToDraw.emplace_back();
			auto &data = mMatchesToDraw.back();

			data.octave = m.getOctave();
			data.position = m.getPositions()[j];
			data.angle = mMapExpander->getTrackerMatchAngle(i);

			bool isReady = data.angle >= mMapExpander->getMinTriangulationAngle();

			if(feature.is3D())
				data.color = StaticColors::Blue(foregroundAlpha);
			else
				data.color = isReady ? StaticColors::Green(foregroundAlpha) : StaticColors::Yellow(foregroundAlpha);
		}
	}
}

void MapExpanderWindow::draw()
{
	if(!mIsInitialized)
		return;

	//Reference image
    mTiler.setActiveTile(0);
    mShaders->getTexture().setMVPMatrix(mTiler.getMVP());
	mShaders->getColor().setMVPMatrix(mTiler.getMVP());

	//Draw texture
	mShaders->getTexture().renderTexture(mCurrentImageTextureTarget, mCurrentImageTextureId, mImageSize);

	//Draw feature coverage mask
	mShaders->getTexture().renderTexture(GL_TEXTURE_2D, mFeatureCoverageTexture.getId(), mImageSize, 0.3f);

	//Prepare text shader
	mShaders->getText().setMVPMatrix(mTiler.getMVP());
	mShaders->getText().setActiveFontSmall();
	mShaders->getText().setRenderCharHeight(5);
	mShaders->getText().setColor(StaticColors::White(1.0f));

	//Draw features to triangulate
	std::vector<std::vector<cv::Point2f>> squareCenters(mOctaveCount);
	std::vector<std::vector<cv::Vec4f>> squareColors(mOctaveCount);
	for(auto it=mMatchesToDraw.begin(),end=mMatchesToDraw.end(); it!=end; ++it)
	{
		auto &data = *it;

		if(data.angle >=0)
		{
			mShaders->getText().setCaret(data.position);
			TextRendererStream(mShaders->getText()) << std::fixed << std::setprecision(3) << data.angle;
		}
		squareCenters[data.octave].push_back(data.position);
		squareColors[data.octave].push_back(data.color);
	}
	//Draw square vectors
	for(int octave=0; octave<mOctaveCount; ++octave)
	{
		int scale = 1<<octave;
		mShaders->getColor().drawRect(squareCenters[octave].data(), squareColors[octave].data(), squareCenters[octave].size(), (float)(scale*PatchWarper::kPatchSize), 1.0f);
	}
}

} /* namespace dtslam */
