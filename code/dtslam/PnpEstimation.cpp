/*
 * PnpEstimation.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "PnpEstimation.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d_c.h> // for CV_P3P
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <ceres/ceres.h>
#include "cvutils.h"
#include "CeresParametrization.h"
#include "CeresUtils.h"
#include "EssentialUtils.h"
#include "Pose3DCeres.h"
#include "CameraModelCeres.h"
#include "ReprojectionError3D.h"
#include "EpipolarSegmentError.h"

#include "flags.h"

namespace dtslam {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PnPRansac
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PnPRansac::PnPRansac()
{
}

PnPRansac::~PnPRansac()
{
}

void PnPRansac::setData(const std::vector<FeatureMatch> *matches, const CameraModel *camera)
{
	assert(matches!=NULL);
	assert(matches->size()!=0);

	mMatchCount = matches->size();

	//Sort 2D and 3D
	for (int i = 0, end = matches->size(); i != end; ++i)
	{
		auto &match = matches->at(i);
		if (match.measurement.getFeature().is3D())
		{
			mMatches3D.push_back(&match);
			mIndexes3D.push_back(i);
		}
		else
		{
			mMatches2D.push_back(&match);
			mIndexes2D.push_back(i);
		}
	}
	int matches3DCount = mMatches3D.size();
	int matches2DCount = mMatches2D.size();

	assert(matches3DCount != 0);

	mConstraintCount = matches3DCount;

	//Normalize
	//TODO: this will break with fish-eye lenses, but cv::triangulate and cv:solvePnP can only take 2D points
	mImageXnNormalized3D.resize(matches3DCount);
	for(int i=0; i!=matches3DCount; ++i)
	{
		const FeatureMatch &match = *mMatches3D[i];
		auto &xns = match.measurement.getPositionXns();
		auto &norm = mImageXnNormalized3D[i];
		int count = xns.size();
		norm.resize(count);
		for(int j=0; j!=count; ++j)
		{
			norm[j] = cvutils::NormalizePoint(xns[j]);
		}

		//Create error functor
		mErrorFunctors3D.emplace_back(new PoseReprojectionError3D(match));
	}

	mRefXnNormalized2D.resize(matches2DCount);
	mImageXnNormalized2D.resize(matches2DCount);
	for(int i=0; i!=matches2DCount; ++i)
	{
		const FeatureMatch &match = *mMatches2D[i];

		mRefXnNormalized2D[i] = cvutils::NormalizePoint(match.sourceMeasurement->getUniquePositionXn());

		auto &norm = mImageXnNormalized2D[i];
		int count = match.measurement.getPositionCount();
		norm.resize(count);
		for(int j=0; j!=count; ++j)
		{
			norm[j] = cvutils::NormalizePoint(match.measurement.getPositionXns()[j]);
		}

		//Create error functor
		mErrorFunctors2D.emplace_back(new EpipolarSegmentErrorForPose(*match.sourceMeasurement, match.measurement, (float)FLAGS_MinDepth));
	}
}

std::vector<FullPose3D> PnPRansac::modelFromMinimalSet(const std::vector<int> &constraintIndices, const std::vector<int> &constraintSubindices)
{
	assert(constraintIndices.size()==4);

	std::vector<cv::Point3f> refp(4);
	std::vector<cv::Point2f> imgp(4);
	for(int i=0; i<4; ++i)
	{
		const int idx = constraintIndices[i];
		const int sidx = constraintSubindices[i];
		const FeatureMatch &match = *mMatches3D[idx];
		refp[i] = match.measurement.getFeature().getPosition();
		imgp[i] = cvutils::NormalizePoint(match.measurement.getPositionXns()[sidx]);
	}

	std::vector<FullPose3D> solutions;

	cv::Vec3d rvec,tvec;
	if(cv::solvePnP(refp, imgp, cv::Matx33f::eye(), cv::noArray(), rvec, tvec, false, CV_P3P))
	{
		cv::Matx33d R;
		cv::Rodrigues(rvec, R);

		solutions.push_back(FullPose3D(R, tvec));
	}
    return solutions;
}

void PnPRansac::getInliers(const FullPose3D &model, int &inlierCount, float &errorSumSq, PnPIterationData &data)
{
	inlierCount = 0;
	errorSumSq = 0;

	ceres::CauchyLoss robustLoss(mOutlierErrorThreshold);

	data.reprojectionErrors.resize(mMatchCount);

	///////////////////////////////////////////
	// 2D matches
	int match2DCount = mErrorFunctors2D.size();
	for(int i=0; i<match2DCount; ++i)
	{
		auto &errorFunctor = *mErrorFunctors2D[i];
		auto &errors = data.reprojectionErrors[mIndexes2D[i]];


		std::vector<float> residuals;
		errorFunctor.computeAllResiduals(model.getRotationRef(), model.getTranslationRef(), residuals);
		CeresUtils::ResidualsToErrors<EpipolarSegmentError::kResidualsPerItem>(errorFunctor.getPointCount(), residuals, mOutlierErrorThresholdSq, errors);
		if(errors.isInlier)
			inlierCount++;

		double robustError[3];
		robustLoss.Evaluate(errors.bestReprojectionErrorSq, robustError);
		errorSumSq += (float)robustError[0];
	}

	///////////////////////////////////////////
	// 3D matches
	int match3DCount = mErrorFunctors3D.size();
	for(int i=0; i<match3DCount; ++i)
	{
		auto &errorFunctor = *mErrorFunctors3D[i];
		auto &errors = data.reprojectionErrors[mIndexes3D[i]];

		errorFunctor.evalToErrors(model.getRotationRef(), model.getTranslationRef(), mOutlierErrorThreshold, errors);

		if(errors.isInlier)
			inlierCount++;

		double robustError[3];
		robustLoss.Evaluate(errors.bestReprojectionErrorSq, robustError);
		errorSumSq += (float)robustError[0];
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PnPRefiner
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool PnPRefiner::getReprojectionErrors3D(const FeatureMatch &match,
		const cv::Matx33f &R,
		const cv::Vec3f &translation,
		MatchReprojectionErrors &errors)
{
	PoseReprojectionError3D err(match);
	err.evalToErrors(R, translation, mOutlierPixelThreshold, errors);
	return errors.isInlier;
}

bool PnPRefiner::getReprojectionErrors2D(const FeatureMatch &match,
		const cv::Matx33f &R,
		const cv::Vec3f &translation,
		MatchReprojectionErrors &errors)
{
	int pcount = match.measurement.getPositionCount();

	EpipolarSegmentErrorForPose err(*match.measurement.getFeature().getMeasurements()[0], match.measurement, (float)FLAGS_MinDepth);
	std::vector<float> residuals(pcount*EpipolarSegmentErrorForPose::kResidualsPerItem);
	err.computeAllResiduals(R, translation, residuals);
	CeresUtils::ResidualsToErrors<EpipolarSegmentErrorForPose::kResidualsPerItem>(pcount, residuals, mOutlierPixelThresholdSq, errors);

	return errors.isInlier;
}

void PnPRefiner::getInliers(const std::vector<FeatureMatch> &matches,
				const cv::Matx33f &R,
				const cv::Vec3f &translation,
				int &inlierCount,
				std::vector<MatchReprojectionErrors> &errors)
{

	inlierCount = 0;

	const int matchCount=matches.size();
	errors.resize(matchCount);
	for(int i=0; i<matchCount; i++)
	{
		auto &match = matches[i];
		auto &error = errors[i];

		if (match.measurement.getFeature().is3D())
		{
			getReprojectionErrors3D(match, R, translation, error);
		}
		else
		{
			getReprojectionErrors2D(match, R, translation, error);
		}
		
		if(error.isInlier)
			inlierCount++;
	}
}

void PnPRefiner::refinePose(const std::vector<FeatureMatch> &matches,
		cv::Matx33f &rotation,
		cv::Vec3f &translation,
		int &inlierCount,
		std::vector<MatchReprojectionErrors> &errors)
{
	const int matchCount=matches.size();

	////////////////////////////////////////////////////
	//Convert input to double

	//Rotation
	const cv::Matx33d rotation_d = rotation;
	cv::Vec3d rparams_d;
	ceres::RotationMatrixToAngleAxis(ceres::RowMajorAdapter3x3(rotation_d.val), rparams_d.val);

	//Transation
	cv::Vec3d translation_d = translation;

	////////////////////////////////////////////////////
	//Check inliers before
	//int inlierCountBefore;
	//getInliers(matches, rparams_d, translation_d, inlierCountBefore, errors);
	//DTSLAM_LOG << "PnPRefine: inliers before: " << inlierCountBefore << "\n";

	int totalMatchCount=0;
	//int refineMatchCount=0;

	////////////////////////////////////////////////////
	// Prepare ceres problem
	//Non-linear minimization with distortion
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	options.preconditioner_type = ceres::SCHUR_JACOBI;
	options.dense_linear_algebra_library_type = ceres::LAPACK;

	options.num_threads = 1; //Multi-threading here adds too much overhead
	options.num_linear_solver_threads = 1;

	options.logging_type = ceres::SILENT;
	options.minimizer_progress_to_stdout = false;

        // in some environments ceres uses std::tr1::shared_ptr, in others
        // it uses std::shared_ptr. Let's keep it simple by not using
        // make_shared.
	options.linear_solver_ordering.reset(new ceres::ParameterBlockOrdering());

	ceres::Problem problem;

	problem.AddParameterBlock(rparams_d.val, 3);
	problem.AddParameterBlock(translation_d.val, 3);
	options.linear_solver_ordering->AddElementToGroup(rparams_d.val, 1);
	options.linear_solver_ordering->AddElementToGroup(translation_d.val, 1);

	for(int i=0; i<matchCount; i++)
	{
		const FeatureMatch &match = matches[i];
		auto &feature = match.measurement.getFeature();

		totalMatchCount++;

		if (feature.is3D())
		{
			//const double costScale = 1;//std::min(10,match.trackLength);
			const int costScale = (feature.getMeasurements().size() > 3) ? 10 : 1;
			//if(costScale <= 1)
			//	continue;

			ceres::LossFunction *lossFuncA = new ceres::CauchyLoss(mOutlierPixelThreshold);
			ceres::LossFunction *lossFunc = new ceres::ScaledLoss(lossFuncA,costScale,ceres::TAKE_OWNERSHIP);

			problem.AddResidualBlock(
					new ceres::AutoDiffCostFunction<PoseReprojectionError3D,2,3,3>(
							new PoseReprojectionError3D(match)),
							lossFunc, rparams_d.val, translation_d.val);
		}
		else if (FLAGS_PoseUse2D)
		{
			//This uses only a single measurement to constrain the 2D feature, not optimal but faster
			auto &refM = *match.measurement.getFeature().getMeasurements()[0];
			
			ceres::LossFunction *lossFunc = new ceres::CauchyLoss(mOutlierPixelThreshold);
			
			problem.AddResidualBlock(
					new ceres::AutoDiffCostFunction<EpipolarSegmentErrorForPose,2,3,3>(
					new EpipolarSegmentErrorForPose(refM, match.measurement, (float)FLAGS_MinDepth)),
					lossFunc, rparams_d.val, translation_d.val);

			////This uses all measurements to constrain the 2D feature. This might give too much weight to the 2D features.
			//for (auto &mPtr : match.measurement.getFeature().getMeasurements())
			//{
			//	auto &m = *mPtr;
			//	if (m.getPositionCount() > 1)
			//		continue;

			//	ceres::LossFunction *lossFunc = new ceres::CauchyLoss(mOutlierPixelThreshold);

			//	problem.AddResidualBlock(
			//		new ceres::AutoDiffCostFunction<EpipolarSegmentErrorForPose, 2, 3, 3>(
			//		new EpipolarSegmentErrorForPose(m, match.measurement, (float)FLAGS_MinDepth)),
			//		lossFunc, rparams_d.val, translation_d.val);
			//	totalMatchCount++;
			//}
		}
	}

	
	////////////////////////////////////////////////////
	// Solve
	ceres::Solver::Summary summary;

	ceres::Solve(options, &problem, &summary);

	//DTSLAM_LOG << summary.FullReport();

	////////////////////////////////////////////////////
	// Again with inliers only
//	ceres::Problem problemInliers;
//
//    options.linear_solver_ordering->Clear();
//
//	problemInliers.AddParameterBlock(rparams_d.val, 3);
//	problemInliers.AddParameterBlock(translation_d.val, 3);
//	options.linear_solver_ordering->AddElementToGroup(rparams_d.val, 1);
//	options.linear_solver_ordering->AddElementToGroup(translation_d.val, 1);
//
//	for(int i=0; i<match3DCount; i++)
//	{
//		const FeatureMatch &match = matches3D[i];
//		//double costScale = std::min(10,match.trackLength);
//		//double costScale = match.trackLength;
//		double costScale = 1;
//		//if(costScale <= 1)
//		//	continue;
//
//		ceres::LossFunction *lossFunc = new ceres::ScaledLoss(NULL,costScale,ceres::TAKE_OWNERSHIP);
//
//		MatchReprojectionErrors errors;
//		if(!getReprojectionErrors3D(match, rparams_d, translation_d, errors))
//			continue;
//
//		problemInliers.AddResidualBlock(
//				new ceres::AutoDiffCostFunction<PoseReprojectionError3D,2,3,3>(
//						new PoseReprojectionError3D(match)),
//				lossFunc, rparams_d.val, translation_d.val);
//		refineMatchCount++;
//	}
//	if(FLAGS_PoseUse2D)
//	{
//		for(int i=0; i<match2DCount; i++)
//		{
//			const FeatureMatch &match = matches2D[i];
//
//			MatchReprojectionErrors errors;
//			if(!getReprojectionErrors2D(match, rparams_d, translation_d, errors))
//				continue;
//
//			for(auto &mPtr : match.measurement.getFeature().getMeasurements())
//			{
//				auto &m = *mPtr;
//				if(m.getPositionCount() > 1)
//					continue;
//
//				problem.AddResidualBlock(
//						new ceres::AutoDiffCostFunction<PoseReprojectionError2D,2,3,3>(
//								new PoseReprojectionError2D(m, match.measurement)),
//						NULL, rparams_d.val, translation_d.val);
//			}
//		}
//		refineMatchCount++;
//	}
//
//	ceres::Solver::Summary summaryInliers;
//	ceres::Solve(options, &problemInliers, &summaryInliers);
//	//DTSLAM_LOG << summaryInliers.FullReport();
//
//	DTSLAM_LOG << "PnPRefine: total matches=" << totalMatchCount << ", inliers used for refinement=" << refineMatchCount << "\n";

	////////////////////////////////////////////////////
	//Extract result
	cv::Matx33d finalRotation_d;
	ceres::AngleAxisToRotationMatrix(rparams_d.val, ceres::RowMajorAdapter3x3(finalRotation_d.val));
	rotation = finalRotation_d;

	translation = translation_d;

	getInliers(matches, rotation, translation, inlierCount, errors);
	DTSLAM_LOG << "PnPRefine: final inlier count=" << inlierCount << "\n";
}

} /* namespace dtslam */
