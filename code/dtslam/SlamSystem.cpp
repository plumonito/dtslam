/*
 * SlamSystem.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 * Contributors : Dawid Pajak (dpajak@nvidia), Rodolfo Schulz de Lima (rodolfos@nvidia.com)
 */

#include "SlamSystem.h"
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <ceres/rotation.h>

#include "SlamKeyFrame.h"
#include "BundleAdjuster.h"
#include "CeresUtils.h"

#include "flags.h"

namespace dtslam
{

const char *SlamSystem::kSavePoseFilename("pose.txt");

bool SlamSystem::init(const CameraModel *camera, double timestamp, cv::Mat3b &imgColor, cv::Mat1b &imgGray)
{
	//Wait for other threads
	if(mBAFuture.valid())
		mBAFuture.get();
	if(mExpanderFuture.valid())
		mExpanderFuture.get();

	//Create first key frame
    std::unique_ptr<SlamKeyFrame> keyFrame(new SlamKeyFrame());
	keyFrame->mOriginalRegionID = 0;
    keyFrame->init(camera, imgColor, imgGray);
    keyFrame->setTimestamp(timestamp);
    keyFrame->setPose(std::unique_ptr<Pose3D>(new FullPose3D()));

    DTSLAM_LOG << "Pyramid sizes: ";
    DTSLAM_LOG << keyFrame->getImage(0).size();
    for(int octave=1; octave<keyFrame->getPyramid().getOctaveCount(); octave++)
    	DTSLAM_LOG << ", " << keyFrame->getImage(octave).size();
    DTSLAM_LOG << "\n";

    DTSLAM_LOG << "SBI size: " << keyFrame->getSBI().size() << "\n";

	//Reset map
	mMap.reset(new SlamMap());
	mPoseLog.clear();

	//Reset tracker
	mTracker.reset(new PoseTracker());
	mTracker->init(camera, keyFrame->getImage(0).size(), keyFrame->getPyramid().getOctaveCount());

	//Prepare map expander
	mMapExpander.reset(new SlamMapExpander());
	mMapExpander->init(camera, this);
	mExpanderCheckPending = false;

	//Start region
	startNewRegion(NULL, std::move(keyFrame));
	mTracker->setActiveRegion(mActiveRegion);
	mTracker->resetTracking(mActiveRegion->getKeyFrames().back()->getPose());

	if(FLAGS_SavePose)
	{
		std::remove(kSavePoseFilename);
	}

	return true;
}

bool SlamSystem::init(const CameraModel *camera, std::unique_ptr<SlamMap> map)
{
	//Wait for other threads
	if (mBAFuture.valid())
		mBAFuture.get();
	if (mExpanderFuture.valid())
		mExpanderFuture.get();

	//Reset map
	mMap = std::move(map);
	mPoseLog.clear();

	//Get a reference keyframe
	if (mMap->getRegions().empty())
		throw std::runtime_error("Cannot init system with a map without regions.");
	mActiveRegion = mMap->getRegions().front().get();
	if (mActiveRegion->getKeyFrames().empty())
		throw std::runtime_error("Cannot init system with a map without keyframes.");
	
	SlamKeyFrame *keyFrame = mMap->getRegions().front()->getKeyFrames().front().get();

	//Reset tracker
	mTracker.reset(new PoseTracker());
	mTracker->init(camera, keyFrame->getImage(0).size(), keyFrame->getPyramid().getOctaveCount());
	mTracker->setActiveRegion(mActiveRegion);
	//mTracker->resetTracking(mActiveRegion->getKeyFrames().back()->getPose());
	mTracker->resetTracking(mActiveRegion->getKeyFrames().front()->getPose());

	//Prepare map expander
	mMapExpander.reset(new SlamMapExpander());
	mMapExpander->init(camera, this);
	mMapExpander->setRegion(mActiveRegion);

	return true;
}

void SlamSystem::startNewRegion(SlamKeyFrame *previousRegionFrame, std::unique_ptr<SlamKeyFrame> keyframe_)
{
	DTSLAM_LOG << "\n-------------------------\nCreating start region\n-------------------------\n";

	//Create region
	SlamRegion *region = mMap->createRegion();

	//Add keyframe
	SlamKeyFrame *keyframe = keyframe_.get();
	region->addKeyFrame(std::move(keyframe_));


	region->setPreviousRegionSourceFrame(previousRegionFrame);

    //Create 2D features
	cv::Size2i imageSize = keyframe->getImage(0).size();
	for (int octave = 0, end = keyframe->getPyramid().getOctaveCount(); octave!=end; ++octave)
    {
    	const int scale = 1<<octave;

    	//FeatureGridIndexer<KeyPointData> keypoints = keyFrame->getKeyPoints(octave).applyNonMaximaSuppresion(scale*PatchWarper::kPatchRightSize);
		//auto &keypoints = keyFrame->getKeyPoints(octave);
		cv::Size2i tileSize(scale*FLAGS_FrameKeypointGridSize, scale*FLAGS_FrameKeypointGridSize);
		auto keypoints = FeatureGridIndexer<KeyPointData>::ApplyNonMaximaSuppresion(keyframe->getKeyPoints(octave), imageSize, tileSize, scale*PatchWarper::kPatchRightSize);
    	for(auto kp : keypoints)
    	{
    		region->createFeature2D(*keyframe, kp.position, kp.xn, octave);
    	}

    	DTSLAM_LOG << "Created " << keypoints.size() << " features in octave " << octave << "\n";
    }
	
    //Set as active region for expander
    mActiveRegion = region;
	mMapExpander->setRegion(mActiveRegion);
}

void SlamSystem::processImage(double timestamp, cv::Mat3b &imgColor, cv::Mat1b &imgGray)
{
	//Reset profiler for each image
	//Profiler::Instance().reset();

	//Build keyframe structure
	std::unique_ptr<SlamKeyFrame> frame(new SlamKeyFrame());
	{
		ProfileSection ss("buildKeyFrame");
		frame->init(&mTracker->getCamera(), imgColor, imgGray);
		frame->setTimestamp(timestamp);
	}

	//Tracking
	bool trackingSuccesful=false;
	{
		//Read lock while tracking
		shared_lock<shared_mutex> lockRead(mMap->getMutex());

		//Reset?
		if (&mTracker->getActiveRegion() != mActiveRegion) //Reset after new region
		{
			DTSLAM_LOG << "Resetting tracker to end of new region.\n";
			mTracker->setActiveRegion(mActiveRegion);
			mTracker->resetTracking(mActiveRegion->getKeyFrames().back()->getPose());
		}

		//Track
		trackingSuccesful = mTracker->trackFrame(std::move(frame));

		//Save pose for evaluation
		if (FLAGS_SavePose)
			saveCurrentPose();
	}

	if (trackingSuccesful)
		mExpanderCheckPending = true;

	SlamKeyFrame *refFrame = mActiveRegion->getKeyFrames().back().get();
	cv::Vec3f poseDiff = mTracker->getCurrentPose().getCenter() - refFrame->getPose().getCenter();
	mPoseLog.emplace_back(refFrame, poseDiff);
}

void SlamSystem::idle()
{
	//Is the expander running?
	bool expanderRunning = isExpanderRunning();
	bool baRunning = isBARunning();

	//Expand map
	if (!expanderRunning && mTracker->getFrame() && mTracker->getPoseType() != EPoseEstimationType::Invalid && mExpanderCheckPending)
	{
		//if (!mActiveRegion->getShouldBundleAdjust()) //This syncs Expander and BA so that we won't add a keyframe until BA is done. Not optimal.
		{
			std::unique_ptr<SlamMapExpander::CheckData> data = std::move(createDataForExpander());
			mExpanderFinished = false;
		
			bool runInline = mSingleThreaded || !mActiveRegion->getFirstTriangulationFrame();
			if (runInline)
			{
				//ProfileSection s("inlineExpander");
				//expanderTask(std::move(data), false);

				//Must run in different thread because of profiler
				mExpanderFuture = std::async(std::launch::async, ExpanderTask, this, data.release(), true);
				mExpanderFuture.wait();
			}
			else
			{
				mExpanderFuture = std::async(std::launch::async, ExpanderTask, this, data.release(), true);
			}

			mExpanderCheckPending = false;
		}
	}

	//BA?
	if (!FLAGS_DisableBA && !mExpanderAdding && !baRunning && mActiveRegion->getShouldBundleAdjust())
	{
		mBAFinished = false;
		mActiveRegion->setAbortBA(false); //BA will set this internally, add here to avoid a false message in the UI

		mBAFuture = std::async(std::launch::async, BundleAdjustTask, this, true);
		if (mSingleThreaded)
		{
			mBAFuture.wait();
		}
	}
}

std::unique_ptr<SlamMapExpander::CheckData> SlamSystem::createDataForExpander()
{
	const auto &frame = *mTracker->getFrame();

	//Add to map
	std::unique_ptr<SlamMapExpander::CheckData> expanderData(new SlamMapExpander::CheckData());

	expanderData->forceAdd = false;
	expanderData->poseType = mTracker->getPoseType();
	expanderData->essentialReferenceFrame = mTracker->getEssentialReferenceFrame();
	expanderData->frame.reset(new SlamKeyFrame(frame)); //Copy frame
	expanderData->frame->setPose(std::unique_ptr<Pose3D>(new FullPose3D(frame.getPose())));

	for(int i=0, end=mTracker->getMatches().size(); i!=end; ++i)
	{
		if(mTracker->getReprojectionErrors()[i].isInlier)
		{
			auto &match = mTracker->getMatches()[i];

			//Create a new FeatureMatch pointing to the duplicated frame
			expanderData->trackedFeatures.push_back(FeatureMatch(match.projection, match.sourceMeasurement, expanderData->frame.get(), match.measurement.getOctave(), match.measurement.getPositions(), match.measurement.getPositionXns(), match.trackLength));
		}
	}

	return expanderData;
}

SlamKeyFrame *SlamSystem::ExpanderTask(SlamSystem *system, SlamMapExpander::CheckData *dataPtr, bool useLocks)
{
	std::unique_ptr<SlamMapExpander::CheckData> data(dataPtr);
	return system->expanderTask(std::move(data), useLocks);
}
SlamKeyFrame *SlamSystem::expanderTask(std::unique_ptr<SlamMapExpander::CheckData> data, bool useLocks)
{
	Profiler::Instance().setCurrentThreadName("expander");

	DTSLAM_LOG << "\nStart expander task...\n";

	const bool doOneFrameBA = !FLAGS_DisableBA;
	bool addFrame = false;

	//Read lock to check
	{
		std::unique_lock<std::mutex> lockLong(mMap->getLongOperationMutex(), std::defer_lock);
		if(useLocks)
			lockLong.lock();

		shared_lock<shared_mutex> lockRead(mMap->getMutex(), std::defer_lock);
		if(useLocks)
			lockRead.lock();

		//Debug: limit total number of key frames
		//if(expander.getRegion()->getKeyFrames().size() >= 2)
		//	return nullptr;
		//DTSLAM_LOG << "Expander: Checking frame\n";
		addFrame = mMapExpander->checkFrame(std::move(data));
		//DTSLAM_LOG << "Expander: Checking frame done\n";
	}
	
	SlamKeyFrame *newFrame = NULL;
	if(addFrame)
	{
		//Prepare one-frame BA
		std::unique_ptr<BundleAdjuster> ba;
		if (doOneFrameBA)
		{
			ba.reset(new BundleAdjuster());
			ba->setOutlierThreshold((float)FLAGS_TrackerOutlierPixelThreshold);
			ba->setUseLocks(useLocks);
			ba->setIsExpanderBA(true);
			ba->setRegion(mMap.get(), mActiveRegion);
			ba->setTracker(mTracker.get());
		}

		{
			std::unique_lock<shared_mutex> lockWrite(mMap->getMutex(), std::defer_lock);
			if(useLocks)
				lockWrite.lock();

			mExpanderAdding = true;
			newFrame = mMapExpander->addKeyFrame();
			if (newFrame)
			{
				//Things to do while map is write-locked
				mActiveRegion->setAbortBA(true);

				if (doOneFrameBA)
				{
					ba->addFrameToAdjust(*newFrame);
				}

				upgradeRelativePoses(newFrame);
			}
		}

		if (newFrame)
		{
			//Things to do without a lock

			//Bundle adjust this frame (BA handles its own locks)
			if (doOneFrameBA)
			{
				mMapExpander->setStatus(ESlamMapExpanderStatus::SingleFrameBA);
				ba->bundleAdjust();
				mMapExpander->setStatus(ESlamMapExpanderStatus::Inactive);
			}

			mActiveRegion->setShouldBundleAdjust(true);
		}

		mExpanderAdding = false;

	}

	DTSLAM_LOG << "\nExpander task done.\n";

	mExpanderFinished = true;
	return newFrame;
}

void SlamSystem::upgradeRelativePoses(const SlamKeyFrame *connectedFrame)
{
	std::unordered_set<SlamKeyFrame *> frames;

	//Gather all connected frames
	for(auto &mref : connectedFrame->getMeasurements())
	{
		auto &feature = mref->getFeature();
		for(auto &m : feature.getMeasurements())
		{
			SlamKeyFrame *frame = &m->getKeyFrame();

			//Skip if same as connected
			if(frame==connectedFrame)
				continue;

			//Skip if pose is already absolute
			const FullPose3D *fullptr = dynamic_cast<const FullPose3D *>(&frame->getPose());
			if(fullptr)
				continue;

			frames.insert(frame);
		}
	}

	//Decide whether we need to upgrade
	for(auto frame : frames)
	{
		int triangulatedCount=0;
		for(auto &m : frame->getMeasurements())
		{
			if(m->getFeature().is3D())
				triangulatedCount++;
		}

		if(triangulatedCount >= FLAGS_ExpanderMinNewTriangulationsForKeyFrame)
		{
			frame->setPose(std::unique_ptr<Pose3D>(new FullPose3D(frame->getPose())));
			DTSLAM_LOG << "Upgraded keyframe " << frame->getTimestamp() << " to full pose.\n";
		}
	}
}

void SlamSystem::BundleAdjustTask(SlamSystem *system, bool useLocks)
{
	system->bundleAdjustTask(useLocks);
}

void SlamSystem::bundleAdjustTask(bool useLocks)
{
	Profiler::Instance().setCurrentThreadName("bundleAdjuster");

	DTSLAM_LOG << "\nStart BA task...\n";

	BundleAdjuster ba;
	ba.setOutlierThreshold((float)FLAGS_TrackerOutlierPixelThreshold);
	ba.setUseLocks(useLocks);
	ba.setIsExpanderBA(false);
	ba.setTracker(mTracker.get());

	{ //Read-lock to prepare BA
		shared_lock<shared_mutex> lockRead(mMap->getMutex(), std::defer_lock);
		if(useLocks)
			lockRead.lock();

		ba.setRegion(mMap.get(), mActiveRegion);
		//for(auto &frame : mActiveRegion->getKeyFrames())
		int adjustFrameCount = FLAGS_GlobalBAFrameCount;
		for (int sz = mActiveRegion->getKeyFrames().size(), i = sz - 1, mn = std::max(0, sz - adjustFrameCount); i >= mn; --i)
		{
			auto &frame = mActiveRegion->getKeyFrames()[i];
			ba.addFrameToAdjust(*frame);
		}
	}

	//BA will lock on its own
	bool success = ba.bundleAdjust();

	if (success)
		mActiveRegion->setShouldBundleAdjust(false);

	DTSLAM_LOG << "\nBA task done...\n";

	mBAFinished = true;
}

void SlamSystem::saveCurrentPose()
{
	if(!mTracker->getFrame() || mTracker->getPoseType() == EPoseEstimationType::Invalid)
		return;

	std::ofstream fs(kSavePoseFilename, std::ios_base::out | std::ios_base::app);
	double currentTime = mTracker->getFrame()->getTimestamp();

	for(auto &frame : mActiveRegion->getKeyFrames())
	{
		saveFramePose(currentTime, frame->getTimestamp(), frame->getPose(), fs);
	}
	saveFramePose(currentTime, mTracker->getFrame()->getTimestamp(), mTracker->getCurrentPose(), fs);
}

void SlamSystem::saveFramePose(double currentTime, double timestamp, const Pose3D &pose, std::ofstream &fs)
{
	fs << currentTime << "," << timestamp << ",";

	cv::Vec3f center = pose.getCenter();
	fs << center[0] << "," << center[1] << "," << center[2] << ",";

	cv::Matx33d R = pose.getRotation();
	cv::Vec3d rparams;
	ceres::RotationMatrixToAngleAxis(CeresUtils::FixedRowMajorAdapter3x3<const double>(R.val),rparams.val);
	fs << rparams[0] << "," << rparams[1] << "," << rparams[2] << "\n";
}

} /* namespace dtslam */
