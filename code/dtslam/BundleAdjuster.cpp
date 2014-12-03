/*
 * BundleAdjuster.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "BundleAdjuster.h"
#include "gflags/gflags.h"

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <opencv2/calib3d.hpp>
#include <ceres/ceres.h>
#include <mutex>
#include "shared_mutex.h"
#include "Profiler.h"
#include "CeresParametrization.h"
#include "ReprojectionError3D.h"
#include "EpipolarSegmentError.h"
#include "Pose3DCeres.h"
#include "PoseTracker.h"

namespace dtslam
{

void BundleAdjuster::addFrameToAdjust(SlamKeyFrame &newFrame)
{
	//Check that it has full pose
	if(dynamic_cast<const FullPose3D*>(&newFrame.getPose()) == nullptr)
	{
		//Invalid frame
		return;
	}

	//Add
	auto itNewFrame=mFramesToAdjust.insert(&newFrame);
	if(itNewFrame.second)
	{
		//New frame!

		//Add features too
		for(auto itM=newFrame.getMeasurements().begin(),endM=newFrame.getMeasurements().end(); itM!=endM; ++itM)
		{
			SlamFeature &feature = (*itM)->getFeature();

			//We need at least two measurements to bundle adjust
			if(feature.getMeasurements().size() > 1)
			{
				mFeaturesToAdjust.insert(&feature);
			}
		}
	}
}

bool BundleAdjuster::isInlier3D(const SlamFeatureMeasurement &measurement, const std::vector<double> &pose, const cv::Vec3d &position)
{
	cv::Vec3d rparams(pose[0],pose[1],pose[2]);
	cv::Vec3d t(pose[3],pose[4],pose[5]);

	MatchReprojectionErrors errors;

	BAReprojectionError3D err(measurement);
	err.evalToErrors(rparams, t, position, mOutlierPixelThreshold, errors);
	return errors.isInlier;
}

bool BundleAdjuster::isInlier2D(const std::pair<SlamFeatureMeasurement,SlamFeatureMeasurement> &measurements, const std::vector<double> &poseFirst, const std::vector<double> &poseSecond)
{
	cv::Vec3d refRparams(poseFirst[0],poseFirst[1],poseFirst[2]);
	cv::Vec3d refT(poseFirst[3],poseFirst[4],poseFirst[5]);
	cv::Vec3d imgRparams(poseSecond[0],poseSecond[1],poseSecond[2]);
	cv::Vec3d imgT(poseSecond[3],poseSecond[4],poseSecond[5]);

	MatchReprojectionErrors errors;

	EpipolarSegmentErrorForBA err(measurements.first, measurements.second, (float)FLAGS_MinDepth);
	std::vector<double> residuals(EpipolarSegmentErrorForBA::kResidualsPerItem*err.getPointCount());
	err.computeAllResiduals(refRparams.val, refT.val, imgRparams.val, imgT.val, residuals.data());
	CeresUtils::ResidualsToErrors<EpipolarSegmentErrorForBA::kResidualsPerItem>(err.getPointCount(), residuals, mOutlierPixelThresholdSq, errors);
	return errors.isInlier;
}

void BundleAdjuster::getInliers(const std::unordered_map<SlamKeyFrame *, std::vector<double>> &paramsPoses,
		const std::unordered_map<SlamFeature *, cv::Vec3d> &paramsFeatures3D,
		const std::vector<SlamFeatureMeasurement> &measurements3D,
		const std::vector<std::pair<SlamFeatureMeasurement,SlamFeatureMeasurement>> &measurements2D,
		int &inlierCount)
{
	inlierCount = 0;
	for(auto &m : measurements3D)
	{
		auto itFeature = paramsFeatures3D.find(&m.getFeature());
		auto itPose = paramsPoses.find(&m.getKeyFrame());
		if(isInlier3D(m, itPose->second, itFeature->second))
			inlierCount++;
	}
	for(auto &m : measurements2D)
	{
		auto itPoseA = paramsPoses.find(&m.first.getKeyFrame());
		auto itPoseB = paramsPoses.find(&m.second.getKeyFrame());
		if(isInlier2D(m, itPoseA->second, itPoseB->second))
			inlierCount++;
	}
}

class BAIterationCallback : public ceres::IterationCallback
{
public:
	BAIterationCallback(const SlamRegion *region) :
		mRegion(region)
	{}

	virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
	{
		if (mRegion->getAbortBA())
			return ceres::SOLVER_ABORT;
		else
			return ceres::SOLVER_CONTINUE;
	}

protected:
	const SlamRegion *mRegion;
};

BundleAdjuster::TGetPoseParamsResult BundleAdjuster::getPoseParams(SlamKeyFrame &frame, std::unordered_map<SlamKeyFrame *, std::vector<double>> &paramsPoses)
{
	auto itNew = paramsPoses.emplace(&frame, std::vector<double>());

	if(itNew.second)
	{
		//Copy pose params
		const Pose3D &pose = frame.getPose();

		auto &params = itNew.first->second;
		params.resize(pose.getDoF());
		pose.copyToArray(params);
	}
	return itNew;
}

bool BundleAdjuster::bundleAdjust()
{
	ProfileSection s("bundleAdjust");

	if(mFramesToAdjust.empty())
		return true;

	//Gather relevant measurements
	std::unordered_map<SlamKeyFrame *, std::vector<double>> paramsPoses;
	std::unordered_map<SlamFeature *, cv::Vec3d> paramsFeatures3D;

	std::vector<SlamFeatureMeasurement> measurements3D;
	std::vector<std::pair<SlamFeatureMeasurement,SlamFeatureMeasurement>> measurements2D;

	//BA ceres problem
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	options.preconditioner_type = ceres::SCHUR_JACOBI;
	options.dense_linear_algebra_library_type = ceres::LAPACK;

	options.num_threads = 4;
	options.num_linear_solver_threads = 4;
	options.logging_type = ceres::SILENT;
	
	options.minimizer_progress_to_stdout = false;

        // in some environments ceres uses std::tr1::shared_ptr, in others
        // it uses std::shared_ptr. Let's keep it simple by not using
        // make_shared.
	options.linear_solver_ordering.reset(new ceres::ParameterBlockOrdering());

	//Abort callback
	std::unique_ptr<BAIterationCallback> callback;
	if (!mIsExpanderBA)
	{
		callback.reset(new BAIterationCallback(mRegion));
		options.callbacks.push_back(callback.get());
	}

	ceres::Problem problem;

	//Reset new key frame flag
	if (!mIsExpanderBA)
		mRegion->setAbortBA(false);

	//Read-lock to prepare ceres problem
	{
		ProfileSection sconstruct("construct");

		shared_lock<shared_mutex> lockRead(mMap->getMutex(), std::defer_lock);
		if(mUseLocks)
			lockRead.lock();

		assert(mRegion);
		assert(mRegion->getKeyFrames().size() >= 2);
		assert(!mFramesToAdjust.empty());
		assert(!mFeaturesToAdjust.empty());

		//Prepare poses
		for(auto &framePtr : mFramesToAdjust)
		{
			SlamKeyFrame &frame = *framePtr;

			const FullPose3D *fullptr = dynamic_cast<const FullPose3D*>(&frame.getPose());
			if(!fullptr)
				continue;

			//Add frame to params list
			auto itNew = getPoseParams(frame, paramsPoses);
			auto &params = itNew.first->second;

			//Add pose as parameter block
			problem.AddParameterBlock(params.data(), 3);
			problem.AddParameterBlock(params.data()+3, 3);
			options.linear_solver_ordering->AddElementToGroup(params.data(), 1);
			options.linear_solver_ordering->AddElementToGroup(params.data()+3, 1);

			if(&frame == mRegion->getKeyFrames().begin()->get())
			{
				//First key frame in region, pose fixed
				problem.SetParameterBlockConstant(params.data());
				problem.SetParameterBlockConstant(params.data()+3);
			}
			else if(&frame == mRegion->getFirstTriangulationFrame())
			{
				//Second key frame, translation norm fixed
				problem.SetParameterization(params.data()+3, new Fixed3DNormParametrization(1.0f));
			}
		}

		//Prepare features
		for(auto &featurePtr : mFeaturesToAdjust)
		{
			SlamFeature &feature = *featurePtr;

			//We need at least two measurements to bundle adjust
			if(feature.getMeasurements().size() > 1)
			{
				if(feature.is3D())
				{
					//Add 3D feature to params list
					auto itNew = paramsFeatures3D.emplace(&feature, cv::Vec3d(cv::Vec3f(feature.getPosition())));
					auto &params = itNew.first->second;

					//Add 3D feature as parameter block
					problem.AddParameterBlock(params.val, 3);
					options.linear_solver_ordering->AddElementToGroup(params.val, 0);

					//Measurements will be added later so that we don't need to find the params again
				}
				else if(FLAGS_PoseUse2D)
				{
					//2D features don't need extra parameters
					SlamFeatureMeasurement &refM = *feature.getMeasurements()[0];

					//Add measurements now
					for(int i=1,end=feature.getMeasurements().size(); i!=end; ++i)
					{
						SlamFeatureMeasurement &m = *feature.getMeasurements()[i];
						measurements2D.push_back(std::make_pair(refM, m)); //Make a copy
					}
				}
			}
		}

		//Add all 3D feautres to ceres problem
		for(auto &params3D : paramsFeatures3D)
		{
			SlamFeature &feature = *params3D.first;
			auto &featureParams = params3D.second;

			//Add all measurements as residual blocks
			for(auto &mPtr : feature.getMeasurements())
			{
				SlamFeatureMeasurement &m = *mPtr;
				SlamKeyFrame &frame = m.getKeyFrame();

				//Ignore if not full pose
				if (dynamic_cast<const FullPose3D*>(&frame.getPose()) == nullptr)
					continue;

				//Make a copy of the measurement
				measurements3D.push_back(m);

				//Get pose
				auto itNewPose = getPoseParams(frame, paramsPoses);
				auto &poseParams = itNewPose.first->second;

				//Is this frame outside of bundle adjustment?
				if(itNewPose.second)
				{
					problem.AddParameterBlock(poseParams.data(), 3);
					problem.AddParameterBlock(poseParams.data()+3, 3);
					options.linear_solver_ordering->AddElementToGroup(poseParams.data(), 1);
					options.linear_solver_ordering->AddElementToGroup(poseParams.data()+3, 1);
					problem.SetParameterBlockConstant(poseParams.data());
					problem.SetParameterBlockConstant(poseParams.data()+3);
				}

				//const int scale = 1<<m.getOctave();
				const double costScale = (feature.getMeasurements().size() > 3) ? 1e6 : 1.0;
				ceres::LossFunction *lossFunc_i = new ceres::CauchyLoss(mOutlierPixelThreshold);
				ceres::LossFunction *scaledLoss = new ceres::ScaledLoss(lossFunc_i, costScale, ceres::TAKE_OWNERSHIP);

				problem.AddResidualBlock(
					new ceres::AutoDiffCostFunction<BAReprojectionError3D,2,3,3,3>(
							new BAReprojectionError3D(m)),
					scaledLoss, poseParams.data(), poseParams.data()+3, featureParams.val);
			}
		}

		//Add all 2D features to ceres problem
		if (FLAGS_PoseUse2D)
		{
			for(auto &m: measurements2D)
			{
				SlamKeyFrame &frameA = m.first.getKeyFrame();
				SlamKeyFrame &frameB = m.second.getKeyFrame();

				//Skip if not full pose
				if (dynamic_cast<const FullPose3D*>(&frameA.getPose()) == nullptr ||
					dynamic_cast<const FullPose3D*>(&frameB.getPose()) == nullptr)
					continue;

				//Get pose A
				auto itNewPoseA = getPoseParams(frameA, paramsPoses);
				auto &poseParamsA = itNewPoseA.first->second;

				//Is this frame outside of bundle adjustment?
				if(itNewPoseA.second)
				{
					problem.AddParameterBlock(poseParamsA.data(), 3);
					problem.AddParameterBlock(poseParamsA.data()+3, 3);
					options.linear_solver_ordering->AddElementToGroup(poseParamsA.data(), 1);
					options.linear_solver_ordering->AddElementToGroup(poseParamsA.data()+3, 1);
					problem.SetParameterBlockConstant(poseParamsA.data());
					problem.SetParameterBlockConstant(poseParamsA.data()+3);
				}

				//Get pose B
				auto itNewPoseB = getPoseParams(frameB, paramsPoses);
				auto &poseParamsB = itNewPoseB.first->second;

				//Is this frame outside of bundle adjustment?
				if(itNewPoseB.second)
				{
					problem.AddParameterBlock(poseParamsB.data(), 3);
					problem.AddParameterBlock(poseParamsB.data()+3, 3);
					options.linear_solver_ordering->AddElementToGroup(poseParamsB.data(), 1);
					options.linear_solver_ordering->AddElementToGroup(poseParamsB.data()+3, 1);
					problem.SetParameterBlockConstant(poseParamsB.data());
					problem.SetParameterBlockConstant(poseParamsB.data()+3);
				}

				//const int costScale = 1<<m.first.getOctave();
				const int costScale = 1;
				ceres::LossFunction *lossFunc_i = new ceres::CauchyLoss(mOutlierPixelThreshold);
				ceres::LossFunction *scaledLoss = new ceres::ScaledLoss(lossFunc_i, costScale, ceres::TAKE_OWNERSHIP);

				problem.AddResidualBlock(
					new ceres::AutoDiffCostFunction<EpipolarSegmentErrorForBA, 2, 3, 3, 3, 3>(
					new EpipolarSegmentErrorForBA(m.first, m.second, (float)FLAGS_MinDepth)),
					scaledLoss, poseParamsA.data(), poseParamsA.data() + 3, poseParamsB.data(), poseParamsB.data() + 3);
			}
		}
	}

	//Get inliers before
	//int inlierCountBefore;
	//getInliers(paramsPoses, paramsFeatures3D, measurements3D, measurements2D, inlierCountBefore);
	//DTSLAM_LOG << "BA inlier count before: " << inlierCountBefore << "\n";

	//No locks while ceres runs
	//Non-linear minimization!
	ceres::Solver::Summary summary;
	{
		ProfileSection ssolve("solve");
		ceres::Solve(options, &problem, &summary);
	}

	if (mIsExpanderBA)
		DTSLAM_LOG << "Expander BA report:\n";
	else
		DTSLAM_LOG << "Main BA report:\n";
	DTSLAM_LOG << summary.FullReport();

	if (summary.termination_type == ceres::USER_FAILURE || (!mIsExpanderBA && mRegion->getAbortBA()))
	{
		DTSLAM_LOG << "\n\nBA aborted due to new key frame in map!!!\n\n";
		return false;
	}
	else if (summary.termination_type == ceres::FAILURE)
	{
		DTSLAM_LOG << "\n\nBA solver failed!!!\n\n" << summary.FullReport();
		return false;
	}
	else
	{
		//Solver finished succesfully
		//Write-lock to update map
		{
			ProfileSection supdate("update");

			std::unique_lock<std::mutex> lockLong(mMap->getLongOperationMutex(), std::defer_lock);
			if(mUseLocks)
				lockLong.lock();

			std::unique_lock<shared_mutex> lockWrite(mMap->getMutex(), std::defer_lock);
			if(mUseLocks)
				lockWrite.lock();

			DTSLAM_LOG << "BA updating map...\n";

			//Update all involved
			//Update frames
			for(auto &paramsPose : paramsPoses)
			{
				SlamKeyFrame &frame = *paramsPose.first;
				auto &params = paramsPose.second;
				frame.getPose().setFromArray(params);
			}

			//Update 3D features
			for(auto &params3D : paramsFeatures3D)
			{
				SlamFeature &feature = *params3D.first;
				auto &featureParams = params3D.second;

				feature.setPosition(cv::Point3f(cv::Vec3f(featureParams)));

				if (!mIsExpanderBA)
				{
					//Update status
					int inlierCount = 0;
					for (auto &mPtr : feature.getMeasurements())
					{
						auto &m = *mPtr;
						auto &pose = m.getKeyFrame().getPose();

						MatchReprojectionErrors errors;
						BAReprojectionError3D err(m);
						err.evalToErrors(pose.getRotation(), pose.getTranslation(), feature.getPosition(), mOutlierPixelThreshold, errors);
						if (errors.isInlier)
							inlierCount++;
					}
					feature.setStatus(inlierCount);

					//Delete?
					if (inlierCount < (int)std::ceil(0.7f*feature.getMeasurements().size()))
					{
						feature.getRegion()->moveToGarbage(feature);
					}
				}
			}

			DTSLAM_LOG << "BA updating done.\n";

			//Mark that we performed BA
			if (mTracker)
				mTracker->resync();
		}

		//Get inliers after
		//int totalCount = measurements3D.size() + measurements2D.size();
		//int inlierCount;
		//getInliers(paramsPoses, paramsFeatures3D, measurements3D, measurements2D, inlierCount);
		//int inlierPercent = (int)(inlierCount*100.0f/totalCount);
		//DTSLAM_LOG << "BA inlier count after: " << inlierCount << " (" << inlierPercent << "%), time: " << summary.total_time_in_seconds << "s\n";

		return true;
	}
}

} /* namespace dtslam */
