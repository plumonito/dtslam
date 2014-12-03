/*
 * SlamSystem.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 * Contributors : Dawid Pajak (dpajak@nvidia), Rodolfo Schulz de Lima (rodolfos@nvidia.com)
 */

#ifndef SLAMSYSTEM_H_
#define SLAMSYSTEM_H_

#include <memory>
#include <future>
#include "stdutils.h"
#include "SlamMap.h"
#include "PoseTracker.h"
#include "SlamMapExpander.h"

namespace dtslam
{

class SlamSystem
{
public:
	SlamSystem(): mSingleThreaded(false) {}

	bool init(const CameraModel *camera, double timestamp, cv::Mat3b &imgColor, cv::Mat1b &imgGray);
	bool init(const CameraModel *camera, std::unique_ptr<SlamMap> map);

	bool isSingleThreaded() const {return mSingleThreaded;}
	void setSingleThreaded(bool value) { mSingleThreaded = value; DTSLAM_LOG << "Set Slam single threaded = " << mSingleThreaded << "\n"; }

	bool isExpanderRunning() {return mExpanderFuture.valid() && !mExpanderFinished;}
	bool isExpanderAdding() const { return mExpanderAdding; }
	bool isBARunning() { return mBAFuture.valid() && !mBAFinished; }

	SlamMap &getMap() {return *mMap;}
	PoseTracker &getTracker() {return *mTracker;}
	SlamMapExpander &getMapExpander() {return *mMapExpander;}

	SlamRegion *getActiveRegion() {return mActiveRegion;}
	void setActiveRegion(SlamRegion *region) {mActiveRegion = region; mMapExpander->setRegion(region);}

	void processImage(double timestamp, cv::Mat3b &imgColor, cv::Mat1b &imgGray);
	
	//Handles thread creation and other maintenance. This should be called when idle and after processImage().
	void idle();

protected:
	////////////////////////////////////////////////////////
	// Members
	static const char *kSavePoseFilename;
	bool mSingleThreaded;

	std::unique_ptr<SlamMap> mMap;
	SlamRegion *mActiveRegion;

	bool mExpanderCheckPending;

	std::unique_ptr<PoseTracker> mTracker;
	std::unique_ptr<SlamMapExpander> mMapExpander;

	std::future<SlamKeyFrame *> mExpanderFuture;
	std::atomic<bool> mExpanderFinished;
	std::atomic<bool> mExpanderAdding;

	std::future<void> mBAFuture;
	std::atomic<bool> mBAFinished;

	////////////////////////////////////////////////////////
	// Methods

	void startNewRegion(SlamKeyFrame *previousRegionFrame, std::unique_ptr<SlamKeyFrame> keyFrame);

	std::unique_ptr<SlamMapExpander::CheckData> createDataForExpander();

	static SlamKeyFrame *ExpanderTask(SlamSystem *system, SlamMapExpander::CheckData *dataPtr, bool useLocks);
	SlamKeyFrame *expanderTask(std::unique_ptr<SlamMapExpander::CheckData> data, bool useLocks);

	void upgradeRelativePoses(const SlamKeyFrame *connectedFrame);

	static void BundleAdjustTask(SlamSystem *system, bool useLocks);
	void bundleAdjustTask(bool useLocks);

	void saveCurrentPose();
	void saveFramePose(double currentTime, double timestamp, const Pose3D &pose, std::ofstream &fs);

public:
	//DEBUG
	std::vector<std::pair<SlamKeyFrame *, cv::Vec3f>> mPoseLog;
};

} /* namespace dtslam */

#endif /* SLAMSYSTEM_H_ */
