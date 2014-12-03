/*
 * MatchesWindow.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "MatchesWindow.h"
#include <opencv2/imgproc.hpp>
#include "dtslam/SlamSystem.h"
#include "dtslam/Pose3DCeres.h"
#include "dtslam/flags.h"
#include "../shaders/DTSlamShaders.h"
#include "../SlamDriver.h"
#include "WindowUtils.h"

#include "dtslam/ReprojectionError3D.h"
#include "dtslam/EpipolarSegmentError.h"

namespace dtslam
{

void MatchesWindow::showHelp() const
{
	BaseWindow::showHelp();
	DTSLAM_LOG << "Shows the results of the FeatureMatcher. The squares represent the succesfully matched features.\n"
			<< "Right click selects a match.\n"
			<< "The square colors represent:\n"
			<< " - Green: 2D inliers\n"
			<< " - Blue: 3D inliers\n"
			<< " - Red: outliers for a feature that has no inliers\n"
			<< " - Yellow: outliers for a feature that has another match that is an inlier\n"
			<< "If you select a specific match, black squares will show all positions that were scanned by the matcher.\n";
}

bool MatchesWindow::init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize)
{
	BaseWindow::init(app, slam, imageSize);

	mCamera = mSlam->getTracker().getCamera();

	mRefFrameTexture.create(GL_RGB, imageSize);

	//Add bindings
	mKeyBindings.addBinding(false,'+',static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&MatchesWindow::increaseDebugMatchIdx),"Cycle over the matches to view details.");
	mKeyBindings.addBinding(false,'-',static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&MatchesWindow::decreaseDebugMatchIdx),"Cycle over the matches to view details.");
	mKeyBindings.addBinding(false,'*',static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&MatchesWindow::resetDebugMatchIdx),"View all matches.");

	resize();
	return true;
}

void MatchesWindow::updateState()
{
	shared_lock<shared_mutex> lockRead(mSlam->getMap().getMutex());

	auto frame = mSlam->getTracker().getFrame();
	if (!frame || mSlam->getTracker().getPoseType() == EPoseEstimationType::Invalid)
	{
		mSelectedMatchAttempt = NULL;
	}
	else
	{
		//Pose
		mFramePose.set(mSlam->getTracker().getCurrentPose());

		//Affine warp
		mProjectionToKeyPointAffine = mSlam->getTracker().getLastToFrameSimilarity();

		// Update selected feature index
		if (!mSelectedFeature)
			mSelectedMatchAttempt = NULL;
		else
		{
			auto it = mSlam->getTracker().getMatchAttempts().find(mSelectedFeature);
			if (it == mSlam->getTracker().getMatchAttempts().end())
				mSelectedMatchAttempt = NULL;
			else
				mSelectedMatchAttempt = &it->second;
		}
	}

	//Calculate errors
	mSelectedErrors = MatchReprojectionErrors();
	if(mSelectedFeature)
	{
		auto match = mSlam->getTracker().getMatch(mSelectedFeature);
		if(match)
		{
			auto &frame = match->measurement.getKeyFrame();

			if (match->projection.getFeature().is3D())
			{
				//3D
				ReprojectionError3D err(match->measurement);
				err.evalToErrors(frame.getPose().getRotation(), frame.getPose().getTranslation(), mSelectedFeature->getPosition(), (float)FLAGS_TrackerOutlierPixelThreshold, mSelectedErrors);
			}
			else
			{
				//2D
				EpipolarSegmentErrorForPose err2(*match->sourceMeasurement, match->measurement, (float)FLAGS_MinDepth);
				std::vector<float> residuals;
				err2.computeAllResiduals(frame.getPose().getRotation(), frame.getPose().getTranslation(), residuals);
				CeresUtils::ResidualsToErrors<EpipolarSegmentErrorForPose::kResidualsPerItem>(err2.getPointCount(), residuals, (float)FLAGS_TrackerOutlierPixelThreshold, mSelectedErrors);
			}
		}
	}

	//////////////////////////////
	// Create drawing stuff
	const MatchReprojectionErrors defaultError;

	const float foregroundAlpha = 0.8f;
	float backgroundAlpha = 0.2f;
	if(!mSelectedMatchAttempt)
		backgroundAlpha = foregroundAlpha;

	//Draw matches
	mSquareCenters.clear();
	mSquareColors.clear();
	mLines.clear();
	mLineColors.clear();
	mEpiLines.clear();
	mDrawText.clear();
	mDrawTextPos.clear();
	mDrawTextColor.clear();
	mTriangleCenters.clear();
	mDrawRefFrame = false;

	//Square centers
	mSquareCenters.resize(mSlam->getTracker().getOctaveCount());
	mSquareColors.resize(mSlam->getTracker().getOctaveCount());

	//Draw projection of selected feature according to previous pose
	if(mSelectedMatchAttempt)
	{
		const FeatureProjectionInfo &projection=mSelectedMatchAttempt->projection;
		const SlamFeature &feature = projection.getFeature();
		const SlamFeatureMeasurement &m = projection.getSourceMeasurement();

		mDrawRefFrame = true;
		mRefFrameTexture.update(m.getKeyFrame().getColorImage());
		mRefMeasurementOctave = m.getOctave();
		mRefMeasurementPos = m.getUniquePosition();

		//Draw candidates
		//Find min score
		int minScore = std::numeric_limits<uint32_t>::max();
		cv::Point2f minPos;

		//Draw all candidate positions
		for(int i=0,end=mSelectedMatchAttempt->candidates.size(); i!=end; ++i)
		{
			auto &candidate = mSelectedMatchAttempt->candidates[i];
			mSquareCenters[projection.getOctave()].push_back(candidate.initialPos);
			mSquareColors[projection.getOctave()].push_back(StaticColors::Gray(foregroundAlpha, 0.25));

			if(candidate.score < minScore)
			{
				minScore = candidate.score;
				minPos = candidate.refinedPos;
			}
		}

		//Show min score
		mSelectedMinScore = (float)minScore;
		//std::stringstream ss;
		//ss << "min score:" << minScore;
		//mDrawText.push_back(ss.str());
		//mDrawTextPos.push_back(minPos);
		//mDrawTextColor.push_back(StaticColors::White(1.0f));

		//Draw
		if(projection.getType() == EProjectionType::EpipolarLine)
		{
			//Draw epipolar line
			auto &epipolarData = projection.getEpipolarData();

			mEpiLines.emplace_back();
			auto &epiLine = mEpiLines.back();

			WindowUtils::BuildEpiLineVertices(frame->getCameraModel(), epipolarData.minDepthXn, epipolarData.infiniteXn, epiLine);
			for(int i=0, end=epiLine.size(); i<end; ++i)
				epiLine[i] = cvutils::AffinePoint(mProjectionToKeyPointAffine, epiLine[i]);
		}
		else
		{
			//Draw triangle at projected pos
			for (auto &pos : projection.getPointData().positions)
			{
				const cv::Point2f posAff = cvutils::AffinePoint(mProjectionToKeyPointAffine, pos);
				mTriangleCenters.push_back(posAff);
			}
		}
	}

	//Draw projection of selected feature according new pose
	mDrawSelectedProjectionTriangle = false;
	mDrawSelectedProjectionEpiLine = false;
	mSelectedProjectionEpiLine.clear();
	if(mSelectedFeature)
	{

		//Calculate projection
		FeatureProjectionInfo projection;
		if(mSelectedFeature->is3D())
		{
			projection = SlamRegion::Project3DFeature(mFramePose, mFramePose.getCenter(), mCamera, mSlam->getTracker().getOctaveCount(), *mSelectedFeature);
		}
		else
		{
			projection = SlamRegion::Project2DFeature(mFramePose, mFramePose.getCenter(), mCamera, *mSelectedFeature->getMeasurements()[0]);
		}

		if (projection.getType() == EProjectionType::PointProjection)
		{
			mDrawSelectedProjectionTriangle = true;
			mSelectedProjectionTriangleCenter = projection.getPointData().positions[0];
		}
		else if (projection.getType() == EProjectionType::EpipolarLine)
		{
			mDrawSelectedProjectionEpiLine = true;
			auto &epipolarData = projection.getEpipolarData();
			WindowUtils::BuildEpiLineVertices(frame->getCameraModel(), epipolarData.minDepthXn, epipolarData.infiniteXn, mSelectedProjectionEpiLine);
		}
	}

	//Draw matches
	for (int i = 0, end = mSlam->getTracker().getMatches().size(); i < end; ++i)
	{
		auto &match = mSlam->getTracker().getMatches()[i];
		const MatchReprojectionErrors &error = mSlam->getTracker().getReprojectionErrors()[i];

		drawMatch(match, error);
	}
}

void MatchesWindow::touchDown(int id, int x, int y)
{
	int tileIdx;
	cv::Vec4f vertex;
	mTiler.screenToVertex(cv::Point2f((float)x, (float)y), tileIdx, vertex);
	cv::Point2f clickPoint(vertex[0], vertex[1]);

	if(id == kMouseRightButton)
	{
		if(tileIdx==0)
		{
			float bestDistSq = std::numeric_limits<float>::infinity();
			for(auto &match : mSlam->getTracker().getMatches())
			{
				for(auto &p : match.measurement.getPositions())
				{
					float distSq = cvutils::PointDistSq(clickPoint, p);
					if(distSq < bestDistSq)
					{
						bestDistSq = distSq;
						mSelectedFeature = &match.measurement.getFeature();
					}
				}

			}

			shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());
			updateState();
		}
	}
}

void MatchesWindow::resize()
{
    mTiler.configDevice(cv::Rect2i(cv::Point2i(0,0),UserInterfaceInfo::Instance().getScreenSize()),1);
	mTiler.fillTiles();
	mTiler.setImageMVP(0, mImageSize);
}

void MatchesWindow::draw()
{
	if(!mIsInitialized)
		return;

	//Reference image
    mTiler.setActiveTile(0);
    mShaders->getTexture().setMVPMatrix(mTiler.getMVP());
	mShaders->getColor().setMVPMatrix(mTiler.getMVP());

	//Draw texture
	mShaders->getTexture().renderTexture(mCurrentImageTextureTarget, mCurrentImageTextureId, mImageSize, 1.0f);

	//Draw square vectors
	glPointSize(3);
	for(int octave=0; octave<(int)mSquareCenters.size(); ++octave)
	{
		int scale = 1<<octave;
		mShaders->getColor().drawRect(mSquareCenters[octave].data(), mSquareColors[octave].data(), mSquareCenters[octave].size(), (float)scale*PatchWarper::kPatchSize,1.0f);
		mShaders->getColor().drawVertices(GL_POINTS, mSquareCenters[octave].data(), mSquareColors[octave].data(), mSquareCenters[octave].size());
	}

	//Draw lines
	for(int i=0,end=mLines.size(); i!=end; ++i)
	{
		mShaders->getColor().drawVertices(GL_LINES, mLines[i].data(), mLines[i].size(), mLineColors[i]);
	}
	//Draw epi lines
	for(int i=0,end=mEpiLines.size(); i!=end; ++i)
	{
		mShaders->getColor().drawVertices(GL_LINE_STRIP, mEpiLines[i].data(), mEpiLines[i].size(), StaticColors::Yellow(0.8f));
	}

	//Draw triangles
	for(int i=0,end=mTriangleCenters.size(); i!=end; ++i)
	{
		const int kRadius = 3;
		const cv::Point2f & c = mTriangleCenters[i];
		cv::Point2f vertices[4] = {cv::Point2f(c.x, c.y+kRadius),
									cv::Point2f(c.x-kRadius, c.y),
									cv::Point2f(c.x+kRadius, c.y),
									cv::Point2f(c.x, c.y-kRadius)};
		mShaders->getColor().drawVertices(GL_TRIANGLE_STRIP, vertices, 4, StaticColors::Yellow(0.8f));
	}

	//Draw selected feature projection
	if(mDrawSelectedProjectionTriangle)
	{
		const int kRadius = 3;
		const cv::Point2f & c = mSelectedProjectionTriangleCenter;
		cv::Point2f vertices[4] = {cv::Point2f(c.x, c.y+kRadius),
									cv::Point2f(c.x-kRadius, c.y),
									cv::Point2f(c.x+kRadius, c.y),
									cv::Point2f(c.x, c.y-kRadius)};
		mShaders->getColor().drawVertices(GL_TRIANGLE_STRIP, vertices, 4, StaticColors::Green(0.8f));
	}

	if(mDrawSelectedProjectionEpiLine)
		mShaders->getColor().drawVertices(GL_LINE_STRIP, mSelectedProjectionEpiLine.data(), mSelectedProjectionEpiLine.size(), StaticColors::Green(0.8f));

	//Draw text
	mShaders->getText().setMVPMatrix(mTiler.getMVP());
	mShaders->getText().setActiveFontSmall();
	mShaders->getText().setRenderCharHeight(10.0f*mImageSize.height/UserInterfaceInfo::Instance().getScreenSize().height);

	for(int i=0,end=mDrawText.size(); i!=end; ++i)
	{
		mShaders->getText().setCaret(mDrawTextPos[i]);
		mShaders->getText().setColor(mDrawTextColor[i]);
		mShaders->getText().renderText(mDrawText[i]);
	}

	//Draw AR cube
    if(mApp->isARCubeValid())
    {
    	std::vector<unsigned int> triangleIndices;
    	std::vector<cv::Vec4f> vertices;
    	std::vector<cv::Vec4f> colors;
    	std::vector<cv::Vec3f> normals;
        mApp->generateARCubeVertices(triangleIndices, vertices, colors, normals);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

		mShaders->getColorCamera().setMVPMatrix(mTiler.getMVP());
		mShaders->getColorCamera().setCamera(mCamera);
		mShaders->getColorCamera().setPose(mFramePose);
		mShaders->getColorCamera().drawVertices(GL_TRIANGLES, triangleIndices.data(), triangleIndices.size(), vertices.data(), colors.data());

	    glDisable(GL_DEPTH_TEST);
    }


    //Switch to full screen
    mTiler.setFullScreen();

	//Ref frame
	if(mDrawRefFrame)
	{
		mShaders->getTexture().setMVPMatrix(mTiler.getMVP());
		mShaders->getColor().setMVPMatrix(mTiler.getMVP());

		//Draw texture
		int height = 100;
		float scale = (float)height / mImageSize.height;
		cv::Size sz((int)(mImageSize.width*scale), height);
		cv::Point2f origin((float)(mTiler.getViewportArea().width-sz.width), (float)(mTiler.getViewportArea().height-sz.height));
		mShaders->getTexture().renderTexture(GL_TEXTURE_2D, mRefFrameTexture.getId(), sz, origin, 1.0f);

		//Ref pos
		cv::Point2f center = origin + scale*mRefMeasurementPos;
		mShaders->getColor().drawRect(&center, 1, StaticColors::Green(), scale*(1<<mRefMeasurementOctave)*PatchWarper::kPatchSize,1);
	}

	//Text
	mShaders->getText().setMVPMatrix(mTiler.getMVP());
	mShaders->getText().setActiveFontBig();
	mShaders->getText().setRenderCharHeight(10);
	mShaders->getText().setCaret(cv::Point2f(300,0));
	mShaders->getText().setColor(StaticColors::Green());

	{
		TextRendererStream ts(mShaders->getText());
		ts << "Matches from tracker\n";

		ts << "Pose inlier count: ";
		int inliers = mSlam->getTracker().getMatchInlierCount();
		int total = mSlam->getTracker().getMatches().size();
		int percentage = (int)((float)inliers/total*100);
		ts << std::setfill('0') << std::setw(3) << inliers
			<< "/" << std::setfill('0') << std::setw(3) << total << " (" << percentage << "%)\n";

		ts << "Debug match: ";
		if(!mSelectedMatchAttempt)
			ts << "all\n";
		else
		{
			ts << "Error: " << sqrtf(mSelectedErrors.bestReprojectionErrorSq);
			if(mSelectedErrors.isInlier)
				ts << "(inlier)";
			else
				ts << "(outlier)";
			ts << "\n";

			ts << "Best matcher score: " << mSelectedMinScore << "\n";

			const FeatureProjectionInfo &projection = mSelectedMatchAttempt->projection;
			ts << "Track length: " << projection.getTrackLength() << "\n";

			const SlamFeatureMeasurement &source = projection.getSourceMeasurement();
			ts << "Source frame: " << source.getKeyFrame().getTimestamp() << "\n";
			ts << "Source position: " << source.getUniquePosition() << "\n";
		}

		switch (mSlam->getTracker().getPoseType())
		{
		case EPoseEstimationType::PureRotation:
			ts << "Rotation\n";
			break;

		case EPoseEstimationType::Essential:
			ts << "Essential\n";
			break;

		case EPoseEstimationType::FullPose:
			ts << "Full motion\n";
			break;

		default:
			ts << "Invalid\n";
			break;
		}
	}
}

void MatchesWindow::drawMatch(const FeatureMatch &match, const MatchReprojectionErrors &errors)
{
	const int octave = match.measurement.getOctave();
	
	cv::Vec4f color = selectMatchColor(match, errors);

	for(int i=0,end=match.measurement.getPositionCount(); i!=end; ++i)
	{
		const cv::Point2f &position = match.measurement.getPositions()[i];
		cv::Vec4f colori;
		if (!errors.isInlier || errors.isImagePointInlier[i])
			colori = color;
		else
			colori = StaticColors::Yellow(color[3]);

		//Store square center to draw
		mSquareCenters[octave].push_back(position);
		mSquareColors[octave].push_back(colori);

		//Draw line from projected to matched position
		std::array<cv::Point2f,2> vertices;
		vertices[0] = position;
		if(match.projection.getType() == EProjectionType::EpipolarLine)
		{
			//Epipolar line, find nearest point on line
			auto &epipolarData = match.projection.getEpipolarData();
			const cv::Point2f posS = cvutils::AffinePoint(mProjectionToKeyPointAffine, position);
			const cv::Point3f xn = mCamera.unprojectToWorld(posS);
			const cv::Point3f xnNearest = xn - cv::Point3f(epipolarData.epiPlaneNormal * epipolarData.epiPlaneNormal.dot(xn));
			const cv::Point2f posNearestS = mCamera.projectFromWorld(xnNearest);
			const cv::Point2f posNearest = cvutils::AffinePoint(mProjectionToKeyPointAffine, posNearestS);

			vertices[1] = posNearest;
		}
		else
		{
			//Normal point projection
			//Find closest point
			cv::Point2f posNearest = match.projection.getPointData().positions[0];
			float minDistSq = cvutils::PointDistSq(position, posNearest);
			for (int i = 1, end = match.projection.getPointData().positions.size(); i != end; ++i)
			{
				const cv::Point2f &projectionPos = match.projection.getPointData().positions[i];
				float distSq = cvutils::PointDistSq(position, projectionPos);
				if(distSq<minDistSq)
				{
					minDistSq = distSq;
					posNearest = projectionPos;
				}
			}
			vertices[1] = cvutils::AffinePoint(mProjectionToKeyPointAffine, posNearest);
		}
		mLines.push_back(vertices);
		mLineColors.push_back(colori);

		//Draw match id text
		if(mSelectedFeature == NULL || mSelectedFeature == &match.measurement.getFeature())
		{
			float error = -1;
			if(i < (int)errors.reprojectionErrorsSq.size())
				error = sqrtf(errors.reprojectionErrorsSq[i]);
			std::stringstream ss;
			ss << std::fixed << std::setprecision(2) << error;
			//ss << match.trackLength;

			//mDrawTextColor.push_back(colori);
			//mDrawTextPos.push_back(position);
			//mDrawText.push_back(ss.str());
		}
	}
}

cv::Vec4f MatchesWindow::selectMatchColor(const FeatureMatch &match, const MatchReprojectionErrors &errors)
{
	const SlamFeature &feature = match.projection.getFeature();
	
	//Determine alpha value
	float alpha;
	{
		const float foregroundAlpha = 0.8f;
		float backgroundAlpha = 0.2f;
		if (!mSelectedMatchAttempt)
			backgroundAlpha = foregroundAlpha;
		alpha = (&feature == mSelectedFeature) ? foregroundAlpha : backgroundAlpha;
	}

	if(!errors.isInlier)
		return StaticColors::Red(alpha); //Outlier
	else if (feature.is3D())
		return StaticColors::Blue(alpha);
	else
		return StaticColors::Green(alpha);
}

void MatchesWindow::increaseDebugMatchIdx()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	auto it = mSlam->getTracker().getMatchAttempts().end();
	if (mSelectedFeature)
	{
		auto it = mSlam->getTracker().getMatchAttempts().find(mSelectedFeature);
		if (it != mSlam->getTracker().getMatchAttempts().end())
			++it;
	}
	else
	{
		it = mSlam->getTracker().getMatchAttempts().begin();
	}

	if (it == mSlam->getTracker().getMatchAttempts().end())
		mSelectedFeature = NULL;
	else
		mSelectedFeature = it->first;

	updateState();
}

void MatchesWindow::decreaseDebugMatchIdx()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	//Solve this with forward iterators because gcc doesn't have backward iterators for unordered_map
	auto &attempts = mSlam->getTracker().getMatchAttempts();
	auto itPrevious = attempts.end();

	if(!attempts.empty())
	{
		auto itSelected = ++attempts.begin();
		itPrevious = attempts.begin();
		//Wrap around, select last feature
		while(itSelected != attempts.end() && itSelected->first != mSelectedFeature) //if mSelectedFeature is NULL this still works
		{
			++itPrevious;
			++itSelected;
		}
	}

	if (itPrevious == attempts.end())
		mSelectedFeature = NULL;
	else
		mSelectedFeature = itPrevious->first;

	updateState();
}

void MatchesWindow::resetDebugMatchIdx()
{
	mSelectedFeature = NULL;

	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());
	updateState();
}

void MatchesWindow::executeRefiner()
{

}

} /* namespace dtslam */
