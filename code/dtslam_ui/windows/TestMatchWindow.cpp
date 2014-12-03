/*
 * TestMatchWindow.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "TestMatchWindow.h"
#include "dtslam/SlamSystem.h"
#include "dtslam/SlamMap.h"
#include "dtslam/SlamKeyFrame.h"
#include "../shaders/DTSlamShaders.h"
#include "dtslam/EssentialEstimation.h"
#include "dtslam/FeatureMatcher.h"
#include "dtslam/MatchRefiner.h"
#include "WindowUtils.h"

#include <opencv2/calib3d.hpp>

namespace dtslam {

bool TestMatchWindow::init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize)
{
	BaseWindow::init(app, slam, imageSize);

	mActiveOctave = mSlam->getTracker().getOctaveCount()-1;
	mClickPointA.x = -1;
	mClickPointB.x = -1;
	mCandidateIdx = -1;
	mRegion = mSlam->getMap().getRegions()[0].get();

	mKeyBindings.addBinding(false, 'q', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TestMatchWindow::nextFrameA), "Select next frame A.");
	mKeyBindings.addBinding(false, 'a', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TestMatchWindow::prevFrameA), "Select previous frame A.");
	mKeyBindings.addBinding(false, 'w', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TestMatchWindow::nextFrameB), "Select next frame B.");
	mKeyBindings.addBinding(false, 's', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TestMatchWindow::prevFrameB), "Select previous frame B.");
	mKeyBindings.addBinding(false, 'l', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TestMatchWindow::nextOctave), "Select next octave.");
	mKeyBindings.addBinding(false, 'c', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TestMatchWindow::nextCandidate), "Select next candidate.");
	mKeyBindings.addBinding(false, 'f', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TestMatchWindow::toggleUseRefiner), "Toggle use of match refiner.");
	mKeyBindings.addBinding(false, 'm', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TestMatchWindow::toggleUseEpipolar), "Toggle matching type (point or epipolar search).");
	mKeyBindings.addBinding(false, 'k', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TestMatchWindow::toggleShowKeyPoints), "Toggle show all key points.");
	mKeyBindings.addBinding(false, '-', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&TestMatchWindow::logPatches), "Write patches to console.");

	mFrameATexture.create(GL_RGB, imageSize);
	mFrameBTexture.create(GL_RGB, imageSize);

	mPatchATexture.create(GL_LUMINANCE, cv::Size(PatchWarper::kPatchSize, PatchWarper::kPatchSize));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    mPatchBTexture.create(GL_LUMINANCE, cv::Size(PatchWarper::kPatchSize, PatchWarper::kPatchSize));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    mValidPatchA = false;
	mValidPatchB = false;

	resize();
	return true;
}

void TestMatchWindow::showHelp() const
{
	BaseWindow::showHelp();
	DTSLAM_LOG << "Tests matching and reports the score.\n";
}

void TestMatchWindow::touchDown(int id, int x, int y)
{
	int tileIdx;
	cv::Vec4f vertex;
	mTiler.screenToVertex(cv::Point2f((float)x, (float)y), tileIdx, vertex);
	cv::Point2f clickPoint(vertex[0], vertex[1]);

	if(id == kMouseLeftButton)
	{
		if(!mValidFrameA || !mValidFrameB)
			return;

		if(tileIdx==0)
		{
			mClickPointA = clickPoint;
		}
		else if(tileIdx==1)
		{
			mClickPointB = clickPoint;
		}

		shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());
		updateState();
	}
}

void TestMatchWindow::resize()
{
    mTiler.configDevice(cv::Rect2i(cv::Point2i(0,0),UserInterfaceInfo::Instance().getScreenSize()),2);
	mTiler.fillTiles();
	mTiler.setImageMVP(0, mImageSize);
	mTiler.setImageMVP(1, mImageSize);
}

void TestMatchWindow::updateState()
{
	shared_lock<shared_mutex> lockRead(mSlam->getMap().getMutex());

	mValidFrameA = false;
	mValidFrameB = false;
	mValidPatchA = false;
	mValidPatchB = false;
	mMatchPosA.x = -1;
	mMatchPosB.x = -1;
	mKeyPointPositions.clear();
	mCandidatePositions.clear();
	mEpiLinesB.clear();

	const SlamKeyFrame *frameA=NULL;
	const SlamKeyFrame *frameB=NULL;
	for(int i=0,end=mRegion->getKeyFrames().size(); i!=end; ++i)
	{
		SlamKeyFrame &frame = *mRegion->getKeyFrames()[i];

		if(i==mFrameAIdx)
		{
			mValidFrameA = true;
			mCameraA = frame.getCameraModel();
			mPoseA.set(frame.getPose());
			frameA = &frame;
		}
		if(i==mFrameBIdx)
		{
			mValidFrameB = true;
			mCameraB = frame.getCameraModel();
			mPoseB.set(frame.getPose());
			frameB = &frame;
		}
	}

	//Handle special case when viewing the tracker frame
	if(mFrameBIdx == -1)
	{
		auto frame = mSlam->getTracker().getFrame();
		if(frame)
		{
			mValidFrameB = true;
			mCameraB = frame->getCameraModel();
			mPoseB.set(frame->getPose());
			frameB = frame;
		}
	}

	//Update textures
	if(mValidFrameA)
		mFrameATexture.update(frameA->getColorImage());
	if(mValidFrameB)
		mFrameBTexture.update(frameB->getColorImage());

	//Determine octave count
	mOctaveCount = 0;
	if(mValidFrameA)
		mOctaveCount = std::max(mOctaveCount, frameA->getPyramid().getOctaveCount());
	else if(mValidFrameB)
		mOctaveCount = std::max(mOctaveCount, frameB->getPyramid().getOctaveCount());

	//Find closest feature to click point A
	if(mValidFrameA)
	{
		float minDistSq = std::numeric_limits<float>::max();
		for(auto &kp : frameA->getKeyPoints(mActiveOctave))
		{
			float distSq = cvutils::PointDistSq(mClickPointA, kp.position);
			if(distSq<minDistSq)
			{
				minDistSq = distSq;
				mMatchPosA = kp.position;
			}
		}
	}

	//Key points
	if(mShowKeyPoints && mValidFrameB)
	{
		for(auto &kp :frameB->getKeyPoints(mActiveOctave))
		{
			mKeyPointPositions.push_back(kp.position);
		}
	}

	//Do match
	//if (mValidFrameA && mValidFrameB && mMatchPosA.x != -1 && mClickPointB.x != -1 && !mUseEpipolar)
	//{
	//	mClickPointB.x = (int)mClickPointB.x;
	//	mClickPointB.y = (int)mClickPointB.y;

	//	//Triangulate
	//	cv::Point3f xnA = frameA->getCameraModel().unprojectToWorld(mMatchPosA);
	//	//cv::Point3f xnB = frameB->getCameraModel().unprojectToWorld(mMatchPosB);
	//	//cv::Matx34f projA = frameA->getPose().getRt();
	//	//cv::Matx34f projB = frameB->getPose().getRt();

	//	//cv::Vec2f xnAnorm = cvutils::NormalizePoint(xnA);
	//	//cv::Vec2f xnBnorm = cvutils::NormalizePoint(xnB);

	//	//DTSLAM_LOG << "projA: " << (cv::Mat)projA << "\n";
	//	//DTSLAM_LOG << "projB: " << (cv::Mat)projB << "\n";
	//	//DTSLAM_LOG << "xnA: " << xnAnorm << "\n";
	//	//DTSLAM_LOG << "xnB: " << xnBnorm << "\n";
	//	//cv::Mat1f p4mat;
	//	//cv::triangulatePoints(projA, projB, xnAnorm, xnBnorm, p4mat);
	//	//cv::Vec4f p4(p4mat(0, 0), p4mat(1, 0), p4mat(2, 0), p4mat(3, 0));

	//	//DTSLAM_LOG << "Triangulated: " << p4 << "\n";

	//	////Plane params
	//	//cv::Vec3f planeNormal = frameA->getPose().getRotation().t() * xnA;
	//	//cv::Point3f planePoint = cv::Point3f(p4[0] / p4[3], p4[1] / p4[3], p4[2] / p4[3]);

	//	//frameA = frameB;
	//	//mClickPointB.x = mMatchPosA.x;
	//	//mClickPointB.y = mMatchPosA.y;

	//	cv::Vec3f planeNormal(0,0,1);
	//	cv::Point3f planePoint(0,0,10);
	//	
	//	FullPose3D poseA(cv::Matx33f::eye(), cv::Vec3f(0, 0, 0));
	//	FullPose3D poseB(cv::Matx33f::eye(), cv::Vec3f(0, 0, 0));
	//	poseA.set(frameA->getPose());
	//	poseB.set(frameB->getPose());

	//	//Testing the 3D patch warper
	//	PatchWarper warper;
	//	warper.setSource(&frameA->getCameraModel(), &frameA->getImage(mActiveOctave), mActiveOctave, mMatchPosA, xnA);
	//	warper.calculateWarp3D(poseA, planeNormal, planePoint, poseB, frameB->getCameraModel(), cv::Matx23f::eye(), mClickPointB, 1<<mActiveOctave);
	//	warper.updatePatch();

	//	mPatchA = warper.getPatch();
	//	mValidPatchA = true;
	//	mPatchATexture.update(mPatchA);
	//	mPatchB = PatchWarper::ExtractPatch(frameB->getImage(mActiveOctave), mClickPointB, mActiveOctave);
	//	mValidPatchB = true;
	//	mPatchBTexture.update(mPatchB);
	//}
	if (mValidFrameA && mValidFrameB && mMatchPosA.x != -1 && (mClickPointB.x != -1 || mUseEpipolar))
	{
		//Build test feature
		cv::Point3f xnA = frameA->getCameraModel().unprojectToWorld(mMatchPosA);
		SlamFeature feature;
		SlamFeatureMeasurement m(&feature, const_cast<SlamKeyFrame*>(frameA), mMatchPosA, xnA, mActiveOctave);

		//Build test projection
		bool validProjection = false;
		FeatureProjectionInfo projection;
		if(mUseEpipolar)
		{
			projection = SlamRegion::Project2DFeature(frameB->getPose(), frameB->getPose().getCenter(), frameB->getCameraModel(), m);
			if (projection.getType() == EProjectionType::EpipolarLine)
			{
				mEpiLinesB.emplace_back();
				WindowUtils::BuildEpiLineVertices(frameB->getCameraModel(), projection.getEpipolarData().minDepthXn, projection.getEpipolarData().infiniteXn, mEpiLinesB.back());
				validProjection = true;
			}
		}
		else
		{
			projection = FeatureProjectionInfo::CreatePoint(&feature, &m, mActiveOctave, mClickPointB);
			validProjection = true;
		}

		if (validProjection)
		{
			FeatureMatcher matcher;
			matcher.setCamera(&frameB->getCameraModel());
			matcher.setSearchDistance(mSlam->getTracker().getMatcherSearchRadius());
			//matcher.setBestScorePercentThreshold(1.3);
			matcher.setMaxZssdScore(50000);
			matcher.setNonMaximaPixelSize(0);
			matcher.setFrame(frameB);
			matcher.setFramePose(frameB->getPose());

			//MatchAttempt attempt;
			std::vector<cv::Point2f> positions;
			std::vector<cv::Point3f> positionXns;
			std::vector<MatchCandidate> candidates;
			mMatchFound = matcher.findMatch(projection,candidates,positions,positionXns);

			//Patch A
			//mPatchA = attempt.refPatch;

			//Store candidate positions
			if(candidates.empty())
			{
				mMatchPosB.x = -1;
				mMatchScore = -1;
			}
			else
			{
				if(mCandidateIdx >= (int)candidates.size())
					mCandidateIdx = -1;

				int idxToShow;
				if(mCandidateIdx==-1)
				{
					int bestScore = candidates[0].score;
					idxToShow = 0;

					for(int i=0,end=candidates.size(); i<end; ++i)
					{
						auto &candidate = candidates[i];
						if(candidate.score < bestScore)
						{
							bestScore = candidate.score;
							idxToShow = i;
						}
					}
				}
				else
					idxToShow = mCandidateIdx;

				auto &candidate = candidates[idxToShow];

				mMatchStartPosB = candidate.initialPos;
				mMatchPosB = candidate.refinedPos;
				mMatchScore = (float)candidate.score;

				//Get patches
				//Patch B
				mPatchB = candidate.refinedPatch;

				//Refine
				if(mUseRefiner)
				{
					int scale = 1<<mActiveOctave;
					float scaleInv = 1.0f/scale;
					MatchRefiner refiner;
					refiner.setRefPatch(&mPatchA);
					refiner.setCenter(mMatchPosB*scaleInv);
					refiner.setImg(&frameB->getImage(mActiveOctave));
					refiner.refine();

					mMatchPosB = refiner.getCenter()*scale;
					mMatchScore = (float)refiner.getScore();
					mPatchB = refiner.getImgPatch();
				}
			}

			//Update textures
			if(!mPatchA.empty())
			{
				mValidPatchA = true;
				mPatchATexture.update(mPatchA);
			}
			if(!mPatchB.empty())
			{
				mValidPatchB = true;
				mPatchBTexture.update(mPatchB);
			}
		}
	}
}

void TestMatchWindow::draw()
{
	int scale = 1<<mActiveOctave;

	//Frame A
	mTiler.setActiveTile(0);
	mShaders->getTexture().setMVPMatrix(mTiler.getMVP());
	mShaders->getColor().setMVPMatrix(mTiler.getMVP());
	if(mValidFrameA)
	{
		mShaders->getTexture().renderTexture(GL_TEXTURE_2D, mFrameATexture.getId(), mFrameATexture.getSize());


		//Draw points
		glPointSize(3);
		mShaders->getColor().drawVertices(GL_POINTS, &mClickPointA, 1, StaticColors::Red());

		//Draw match
		if(mMatchPosA.x != -1)
		{
			mShaders->getColor().drawRect(&mMatchPosA, 1, StaticColors::Green(), (float)(scale*PatchWarper::kPatchSize), 1.0f);
		}

		//Draw patch
		if(mValidPatchA)
		{
			cv::Size sz(10*mPatchBTexture.getSize().width, 10*mPatchBTexture.getSize().height);
			mShaders->getTexture().renderTexture(GL_TEXTURE_2D, mPatchATexture.getId(), sz);
		}
	}

	//Frame B
	mTiler.setActiveTile(1);
	mShaders->getTexture().setMVPMatrix(mTiler.getMVP());
	mShaders->getColor().setMVPMatrix(mTiler.getMVP());
	if(mValidFrameB)
	{
		mShaders->getTexture().renderTexture(GL_TEXTURE_2D, mFrameBTexture.getId(), mFrameBTexture.getSize());

		//Key points
		if(mShowKeyPoints)
		{
			mShaders->getColor().drawRect(mKeyPointPositions.data(), mKeyPointPositions.size(), StaticColors::Yellow(0.3f), (float)(scale*PatchWarper::kPatchSize), 1);
		}

		//Candidates
		mShaders->getColor().drawRect(mCandidatePositions.data(), mCandidatePositions.size(), StaticColors::Black(0.3f), (float)(scale*PatchWarper::kPatchSize), 1);

		//Epiline
		for(int i=0,end=mEpiLinesB.size(); i!=end; ++i)
			mShaders->getColor().drawVertices(GL_LINE_STRIP, mEpiLinesB[i].data(), mEpiLinesB[i].size(), StaticColors::Yellow());

		//Draw points
		glPointSize(3);
		mShaders->getColor().drawVertices(GL_POINTS, &mClickPointB, 1, StaticColors::Red());

		//Draw match
		if(mMatchPosB.x != -1)
		{
			mShaders->getColor().drawRect(&mMatchStartPosB, 1, StaticColors::Yellow(), (float)(scale*PatchWarper::kPatchSize), 1);
			mShaders->getColor().drawRect(&mMatchPosB, 1, StaticColors::Green(), (float)(scale*PatchWarper::kPatchSize), 1);

			//Draw text
			mShaders->getText().setMVPMatrix(mTiler.getMVP());
			mShaders->getText().setActiveFontSmall();
			mShaders->getText().setRenderCharHeight(10);
			mShaders->getText().setCaret(mMatchPosB);
			mShaders->getText().setColor(StaticColors::White());

			TextRendererStream ts(mShaders->getText());
			ts << mMatchScore;
		}

		//Draw patch
		if(mValidPatchB)
		{
			cv::Size sz(10*mPatchBTexture.getSize().width, 10*mPatchBTexture.getSize().height);
			mShaders->getTexture().renderTexture(GL_TEXTURE_2D, mPatchBTexture.getId(), sz);
		}
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

		ts << "Frames " << mFrameAIdx << " and ";
		if(mFrameBIdx==-1)
			ts << "[tracker]";
		else
			ts << mFrameBIdx;
		ts << "\n";

		ts << "Octave " << mActiveOctave << "\n";

		ts << "Match mode: ";
		if(mUseEpipolar)
			ts << "epipolar";
		else
			ts << "point";
		ts << "\n";

		ts << "Use refiner: ";
		if(mUseRefiner)
			ts << "yes";
		else
			ts << "no ";
		ts << "\n";

		ts << "Candidate: ";
		if(mCandidateIdx == -1)
			ts << "best\n";
		else
			ts << mCandidateIdx << "\n";

		ts << "Score: " << mMatchScore << "\n";
		ts << "Click point A: " << mClickPointA << "\n";
		ts << "Click point B: " << mClickPointB << "\n";
		ts << "Match position A: " << mMatchPosA << "\n";
		ts << "Match start position B: " << mMatchStartPosB << "\n";
		ts << "Match refined position B: " << mMatchPosB << "\n";
	}
}

void TestMatchWindow::nextFrameA()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mFrameAIdx++;
	if(mFrameAIdx>=(int)mRegion->getKeyFrames().size())
		mFrameAIdx = 0;

	updateState();
}
void TestMatchWindow::prevFrameA()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mFrameAIdx--;
	if(mFrameAIdx<0)
		mFrameAIdx = (int)mRegion->getKeyFrames().size()-1;

	updateState();
}
void TestMatchWindow::nextFrameB()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mFrameBIdx++;
	if(mFrameBIdx>=(int)mRegion->getKeyFrames().size())
		mFrameBIdx = -1;

	updateState();
}
void TestMatchWindow::prevFrameB()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mFrameBIdx--;
	if(mFrameBIdx<-1)
		mFrameBIdx = (int)mRegion->getKeyFrames().size()-1;
	updateState();
}

void TestMatchWindow::nextOctave()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mCandidateIdx = -1;
	mActiveOctave++;
	if(mActiveOctave >= mOctaveCount)
		mActiveOctave = 0;
	updateState();
}

void TestMatchWindow::nextCandidate()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mCandidateIdx++;
	updateState();
}

void TestMatchWindow::toggleUseRefiner()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mUseRefiner = !mUseRefiner;
	updateState();
}

void TestMatchWindow::toggleUseEpipolar()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mUseEpipolar = !mUseEpipolar;
	updateState();
}

void TestMatchWindow::toggleShowKeyPoints()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	mShowKeyPoints = !mShowKeyPoints;
	updateState();
}

void TestMatchWindow::logPatches()
{
	if(mValidPatchA)
		DTSLAM_LOG << "patchA=" << mPatchA << "\n";
	if(mValidPatchB)
		DTSLAM_LOG << "patchB=" << mPatchB << "\n";
}

} /* namespace dtslam */
