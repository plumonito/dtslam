/*
 * KeyFramePairWindow.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "KeyFramePairWindow.h"
#include "dtslam/SlamSystem.h"
#include "dtslam/SlamMap.h"
#include "dtslam/SlamKeyFrame.h"
#include "../shaders/DTSlamShaders.h"
#include "dtslam/EssentialEstimation.h"
#include "WindowUtils.h"

namespace dtslam {

bool KeyFramePairWindow::init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize)
{
	TwoFrameWindow::init(app, slam, imageSize);

	mShowKeyPoints = false;

	mKeyBindings.addBinding(false, 'q', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&KeyFramePairWindow::nextFrameA), "Select next frame A.");
	mKeyBindings.addBinding(false, 'a', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&KeyFramePairWindow::prevFrameA), "Select previous frame A.");
	mKeyBindings.addBinding(false, 'w', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&KeyFramePairWindow::nextFrameB), "Select next frame B.");
	mKeyBindings.addBinding(false, 's', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&KeyFramePairWindow::prevFrameB), "Select previous frame B.");
	mKeyBindings.addBinding(false, '+', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&KeyFramePairWindow::nextMeasurement), "Select next feature.");
	mKeyBindings.addBinding(false, '-', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&KeyFramePairWindow::prevMeasurement), "Select previous feature.");
	mKeyBindings.addBinding(false, '*', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&KeyFramePairWindow::allMeasurements), "Select no feature.");
	mKeyBindings.addBinding(false, 'l', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&KeyFramePairWindow::nextOctave), "Select next octave.");
	mKeyBindings.addBinding(false, 'k', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&KeyFramePairWindow::toggleShowKeyPoints), "Show key points.");

	mFrameATexture.create(GL_RGB, imageSize);
	mFrameBTexture.create(GL_RGB, imageSize);

	resize();
	return true;
}

void KeyFramePairWindow::showHelp() const
{
	TwoFrameWindow::showHelp();
	DTSLAM_LOG << "Shows the matches that are common between two key frames.\n"
			<< " Left click: show epipolar line for this specicic point.\n"
			<< " Right click: select nearest feature.\n";
}

void KeyFramePairWindow::touchDown(int id, int x, int y)
{
	int tileIdx;
	cv::Vec4f vertex;
	mTiler.screenToVertex(cv::Point2f((float)x, (float)y), tileIdx, vertex);
	cv::Point2f clickPoint(vertex[0], vertex[1]);

	if(id == kMouseLeftButton)
	{
		if(!mValidFrameA || !mValidFrameB)
			return;

		mEpiLinesA.clear();
		mEpiLinesB.clear();

		if(tileIdx==0)
		{
			mClickPointsA.push_back(clickPoint);
		}
		else if(tileIdx==1)
		{
			mClickPointsB.push_back(clickPoint);
		}

		shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());
		updateSelectedMeasurement();
	}
	else if(id == kMouseRightButton)
	{
		if(tileIdx==0)
		{
			float bestDistSq = std::numeric_limits<float>::infinity();
			for(auto &ma : mMeasurementsA)
			{
				for(auto &p : ma.getPositions())
				{
					float distSq = cvutils::PointDistSq(p, clickPoint);

					if(distSq < bestDistSq)
					{
						bestDistSq = distSq;
						mSelectedFeature = &ma.getFeature();
					}
				}
			}
		}
		else if(tileIdx==1)
		{
			float bestDistSq = std::numeric_limits<float>::infinity();
			for(auto &mb : mMeasurementsB)
			{
				for(auto &p : mb.getPositions())
				{
					float distSq = cvutils::PointDistSq(p, clickPoint);

					if(distSq < bestDistSq)
					{
						bestDistSq = distSq;
						mSelectedFeature = &mb.getFeature();
					}
				}
			}
		}

		shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());
		updateSelectedMeasurement();
	}
}

void KeyFramePairWindow::updateState(const SlamKeyFrame &frameA, const SlamKeyFrame &frameB)
{
	mMeasurementsA.clear();
	mMeasurementsB.clear();

	mCameraA = frameA.getCameraModel();
	mPoseA = frameA.getPose();
	mCameraB = frameB.getCameraModel();
	mPoseB = frameB.getPose();

	//Determine octave count
	mOctaveCount = std::max(mOctaveCount, frameA.getPyramid().getOctaveCount());

	//Update measurements
	if(mShowKeyPoints)
	{
		const int minOctave = (mActiveOctave==-1) ? 0 : mActiveOctave;
		const int endOctave = (mActiveOctave==-1) ? mOctaveCount : mActiveOctave+1;
		for(int octave=minOctave; octave!=endOctave; ++octave)
		{
			//Add key points for frame A
			for(auto &kp : frameA.getKeyPoints(octave))
			{
				mMeasurementsA.push_back(SlamFeatureMeasurement(NULL, NULL, kp.position, kp.xn, octave));
			}

			//Add key points for frame B
			for(auto &kp : frameB.getKeyPoints(octave))
			{
				mMeasurementsB.push_back(SlamFeatureMeasurement(NULL, NULL, kp.position, kp.xn, octave));
			}
		}
	}
	else
	{
		if(mFrameBIdx != -1)
		{
			//Normal key frame for B
			for(auto itA=frameA.getMeasurements().begin(), endA=frameA.getMeasurements().end(); itA!=endA; ++itA)
			{
				SlamFeatureMeasurement &ma = **itA;
				if(mActiveOctave != -1 && ma.getOctave() != mActiveOctave)
					continue;

				for(auto itB=frameB.getMeasurements().begin(), endB=frameB.getMeasurements().end(); itB!=endB; ++itB)
				{
					SlamFeatureMeasurement &mb = **itB;
					if(&ma.getFeature() == &mb.getFeature())
					{
						mMeasurementsA.push_back(ma);
						mMeasurementsB.push_back(mb);
						break;
					}
				}
			}
		}
		else
		{
			//Tracker frame for B

			//Outer loop over features in frame A
			for(auto &map : frameA.getMeasurements())
			{
				SlamFeatureMeasurement &ma = *map;

				//Skip non-active octaves
				if(mActiveOctave != -1 && ma.getOctave() != mActiveOctave)
					continue;

				auto match = mSlam->getTracker().getMatch(&ma.getFeature());
				if (match)
				{
					//Add measurements!
					mMeasurementsA.push_back(ma);
					mMeasurementsB.push_back(match->measurement);
				}
			}
		}
	}

	if(mSelectedMeasurementIdxA >= (int)mMeasurementsA.size())
		mSelectedMeasurementIdxA = -1;
	updateSelectedMeasurement();

	//Text
	if(mActiveOctave == -1)
		mDisplayText << "All octaves";
	else
		mDisplayText << "Octave " << mActiveOctave;
	mDisplayText << "\n";

	mDisplayText << "Shared measurements: " << mMeasurementsA.size() << ", " << mMeasurementsB.size() << "\n";
}

void KeyFramePairWindow::draw()
{
	TwoFrameWindow::draw();

	//Frame A
	mTiler.setActiveTile(0);
	mShaders->getTexture().setMVPMatrix(mTiler.getMVP());
	mShaders->getColor().setMVPMatrix(mTiler.getMVP());
	if(mValidFrameA)
	{
		drawMeasurements(mMeasurementsA);

		for(int i=0,end=mEpiLinesA.size(); i!=end; ++i)
			mShaders->getColor().drawVertices(GL_LINE_STRIP, mEpiLinesA[i].data(), mEpiLinesA[i].size(), StaticColors::Yellow());

		//Draw points
		glPointSize(3);
		mShaders->getColor().drawVertices(GL_POINTS, mClickPointsA.data(), mClickPointsA.size(), StaticColors::Red());

		if (mValidSelectedProjection)
			mShaders->getColor().drawVertices(GL_POINTS, &mSelectedProjectionA, 1, StaticColors::Green());
	}

	//Frame B
	mTiler.setActiveTile(1);
	mShaders->getTexture().setMVPMatrix(mTiler.getMVP());
	mShaders->getColor().setMVPMatrix(mTiler.getMVP());
	if(mValidFrameB)
	{
		drawMeasurements(mMeasurementsB);

		for(int i=0,end=mEpiLinesB.size(); i!=end; ++i)
			mShaders->getColor().drawVertices(GL_LINE_STRIP, mEpiLinesB[i].data(), mEpiLinesB[i].size(), StaticColors::Yellow());

		//Draw points
		glPointSize(3);
		mShaders->getColor().drawVertices(GL_POINTS, mClickPointsB.data(), mClickPointsB.size(), StaticColors::Red());

		if (mValidSelectedProjection)
			mShaders->getColor().drawVertices(GL_POINTS, &mSelectedProjectionB, 1, StaticColors::Green());
	}
}

void KeyFramePairWindow::drawMeasurements(std::vector<SlamFeatureMeasurement> &measurements)
{
	const float foregroundAlpha = 0.8f;
	float backgroundAlpha = 0.2f;
	if(mSelectedFeature==NULL)
		backgroundAlpha = foregroundAlpha;

	std::vector<std::vector<cv::Point2f>> squareCenters(mOctaveCount);
	std::vector<std::vector<cv::Vec4f>> squareColors(mOctaveCount);
	for(int i=0,end=measurements.size(); i!=end; ++i)
	{
		SlamFeatureMeasurement &m = measurements[i];
		SlamFeature *feature = &m.getFeature();

		cv::Vec4f color;
		if(mSelectedFeature == &m.getFeature())
			color = StaticColors::Red(foregroundAlpha);
		else if(feature && feature->is3D())
			color = StaticColors::Blue(backgroundAlpha);
		else
			color = StaticColors::Green(backgroundAlpha);

		for(auto &pos : m.getPositions())
		{
			squareCenters[m.getOctave()].push_back(pos);
			squareColors[m.getOctave()].push_back(color);
		}
	}
	//Draw square vectors
	glPointSize(3);
	for(int octave=0; octave<(int)squareCenters.size(); ++octave)
	{
		int scale = 1<<octave;
		mShaders->getColor().drawRect(squareCenters[octave].data(), squareColors[octave].data(), squareCenters[octave].size(), (float)(scale*PatchWarper::kPatchSize),1.0f);
		mShaders->getColor().drawVertices(GL_POINTS, squareCenters[octave].data(), squareColors[octave].data(), squareCenters[octave].size());
	}
}


void KeyFramePairWindow::nextMeasurement()
{
	mSelectedMeasurementIdxA++;
	if(mSelectedMeasurementIdxA>=(int)mMeasurementsA.size())
		mSelectedFeature = NULL;
	else
		mSelectedFeature = &mMeasurementsA[mSelectedMeasurementIdxA].getFeature();
	updateSelectedMeasurement();
}
void KeyFramePairWindow::prevMeasurement()
{
	mSelectedMeasurementIdxA--;
	if(mSelectedMeasurementIdxA==-1)
		mSelectedFeature = NULL;
	else if(mSelectedMeasurementIdxA<-1 && !mMeasurementsA.empty())
		mSelectedFeature = &mMeasurementsA.back().getFeature();
	else
		mSelectedFeature = &mMeasurementsA[mSelectedMeasurementIdxA].getFeature();
	updateSelectedMeasurement();
}

void KeyFramePairWindow::allMeasurements()
{
	mSelectedFeature = NULL;
	mClickPointsA.clear();
	mClickPointsB.clear();
	updateSelectedMeasurement();
}

void KeyFramePairWindow::updateSelectedMeasurement()
{
	mEpiLinesA.clear();
	mEpiLinesB.clear();
	mValidPatchA = false;
	mValidPatchB = false;
	mValidSelectedProjection = false;

	if(!mValidFrameA || !mValidFrameB)
	{
		mSelectedMeasurementIdxA = -1;
		return;
	}

	if(mSelectedFeature)
	{
		//Update index A
		SlamFeatureMeasurement *ma = NULL;
		for(int i=0,end=mMeasurementsA.size(); i!=end; ++i)
		{
			SlamFeatureMeasurement &mai = mMeasurementsA[i];
			if(&mai.getFeature() == mSelectedFeature)
			{
				mSelectedMeasurementIdxA = i;
				ma = &mai;
				break;
			}
		}

		//Update index B
		SlamFeatureMeasurement * mb = NULL;
		for(int i=0,end=mMeasurementsB.size(); i!=end; ++i)
		{
			SlamFeatureMeasurement &mbi = mMeasurementsB[i];
			if(&mbi.getFeature() == mSelectedFeature)
			{
				mb = &mbi;
				break;
			}
		}

		//Get 3D projection
		if (mSelectedFeature->is3D())
		{
			mValidSelectedProjection = true;
			if (ma)
				mSelectedProjectionA = ma->getCamera().projectFromWorld(ma->getFramePose().apply(mSelectedFeature->getPosition()));
			if (mb)
			mSelectedProjectionB = mb->getCamera().projectFromWorld(mb->getFramePose().apply(mSelectedFeature->getPosition()));
		}

		//Do we have valid measurements on both images?
		if(ma && mb)
		{
			//Project onto image A
			for(int i=0,end=mb->getPositionCount(); i!=end; ++i)
			{
				EpipolarProjection projectionA;
				projectionA = SlamRegion::CreateEpipolarProjection(mPoseB, mb->getPositionXns()[i], mPoseA);
				mEpiLinesA.emplace_back();
				WindowUtils::BuildEpiLineVertices(mCameraA, projectionA.minDepthXn, projectionA.infiniteXn, mEpiLinesA.back());
			}

			//Project onto image B
			for(int i=0,end=ma->getPositionCount(); i!=end; ++i)
			{
				EpipolarProjection projectionB;
				projectionB = SlamRegion::CreateEpipolarProjection(mPoseA, ma->getPositionXns()[i], mPoseB);
				mEpiLinesB.emplace_back();
				WindowUtils::BuildEpiLineVertices(mCameraB, projectionB.minDepthXn, projectionB.infiniteXn, mEpiLinesB.back());
			}
//
//			//Update epipolar lines
//			cv::Vec3f centerA = -mPoseA.getRotationRef().t() * mPoseA.getTranslationRef();
//			cv::Vec3f centerB = -mPoseB.getRotationRef().t() * mPoseB.getTranslationRef();
//
//			cv::Point3f epipoleA = mPoseA.apply(centerB);
//			for(auto &xnB : mb->getPositionXns())
//			{
//				cv::Point3f infiniteXnA = mPoseA.getRotationRef()*mPoseB.getRotationRef().t()*xnB;
//				mEpiLinesA.emplace_back();
//				WindowUtils::BuildEpiLineVertices(mCameraA, epipoleA, infiniteXnA, mEpiLinesA.back());
//			}
//
//			cv::Point3f epipoleB = mPoseB.apply(centerA);
//			for(auto &xnA : ma->getPositionXns())
//			{
//				cv::Point3f infiniteXnB = mPoseB.getRotationRef()*mPoseA.getRotationRef().t()*xnA;
//				mEpiLinesB.emplace_back();
//				WindowUtils::BuildEpiLineVertices(mCameraB, epipoleB, infiniteXnB, mEpiLinesB.back());
//			}

			//Update patches

		}
	}

	//Add epipolar lines of clicked points
	for(auto &pointA : mClickPointsA)
	{
//		//Build epi line b
//		cv::Vec3f centerA = -mPoseA.getRotationRef().t() * mPoseA.getTranslationRef();
//		cv::Point3f epipoleB = mPoseB.apply(centerA);
//		cv::Point3f infiniteXnB = mPoseB.getRotationRef()*mPoseA.getRotationRef().t()*mCameraA.unprojectToWorld(pointA);
//		mEpiLinesB.emplace_back();
//		WindowUtils::BuildEpiLineVertices(mCameraB, epipoleB, infiniteXnB, mEpiLinesB.back());

		EpipolarProjection projectionB;
		projectionB = SlamRegion::CreateEpipolarProjection(mPoseA, mCameraA.unprojectToWorld(pointA), mPoseB);
		mEpiLinesB.emplace_back();
		WindowUtils::BuildEpiLineVertices(mCameraB, projectionB.minDepthXn, projectionB.infiniteXn, mEpiLinesB.back());
	}
	for(auto &pointB : mClickPointsB)
	{
//		//Build epi line a
//		cv::Vec3f centerB = -mPoseB.getRotationRef().t() * mPoseB.getTranslationRef();
//		cv::Point3f epipoleA = mPoseA.apply(centerB);
//		cv::Point3f infiniteXnA = mPoseA.getRotationRef()*mPoseB.getRotationRef().t()*mCameraB.unprojectToWorld(pointB);
//		mEpiLinesA.emplace_back();
//		WindowUtils::BuildEpiLineVertices(mCameraA, epipoleA, infiniteXnA, mEpiLinesA.back());

		EpipolarProjection projectionA;
		projectionA = SlamRegion::CreateEpipolarProjection(mPoseB, mCameraB.unprojectToWorld(pointB), mPoseA);
		mEpiLinesA.emplace_back();
		WindowUtils::BuildEpiLineVertices(mCameraA, projectionA.minDepthXn, projectionA.infiniteXn, mEpiLinesA.back());
	}
}

void KeyFramePairWindow::nextOctave()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mActiveOctave++;
	if(mActiveOctave >= mOctaveCount)
		mActiveOctave = -1;
	TwoFrameWindow::updateState();
}

void KeyFramePairWindow::toggleShowKeyPoints()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mShowKeyPoints=!mShowKeyPoints;
	TwoFrameWindow::updateState();
}
} /* namespace dtslam */
