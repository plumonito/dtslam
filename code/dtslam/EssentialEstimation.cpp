/*
 * EssentialEstimation.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "EssentialEstimation.h"
#include <cassert>

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/calib3d.hpp>

#include "EssentialUtils.h"
#include "CeresParametrization.h"
#include "cvutils.h"
#include "CeresUtils.h"
#include "Pose3DCeres.h"
#include "EpipolarSegmentError.h"

namespace dtslam {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EssentialRansac
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
EssentialRansac::EssentialRansac()
{
}
EssentialRansac::~EssentialRansac()
{
}

void EssentialRansac::setData(const CameraModel *camera, const std::vector<FeatureMatch> *referenceFrameMatches, const std::vector<FeatureMatch> *allMatches)
{
	assert(camera);
	assert(referenceFrameMatches);
	assert(allMatches);
	mCamera = camera;
	mReferenceFrameMatches = referenceFrameMatches;
	mAllMatches = allMatches;

	//Copy pose of reference frame
	mReferenceFramePose = mReferenceFrameMatches->at(0).sourceMeasurement->getKeyFrame().getPose();

	mConstraintCount = mReferenceFrameMatches->size();
	assert(mConstraintCount);

	//Normalize
	mRefXnPointsNormalized.resize(mConstraintCount);
	mImageXnPointsNormalized.resize(mConstraintCount);
	for(int i=0; i!=mConstraintCount; ++i)
	{
		auto &match = mReferenceFrameMatches->at(i);
		auto &refXn = match.sourceMeasurement->getUniquePositionXn();

		mRefXnPointsNormalized[i] = cvutils::NormalizePoint(refXn);

		auto &pointsN = mImageXnPointsNormalized[i];
		pointsN.resize(match.measurement.getPositionCount());
		for(int j=0,endj=match.measurement.getPositionCount(); j!=endj; ++j)
			pointsN[j] = cvutils::NormalizePoint(match.measurement.getPositionXns()[j]);
	}

	for(auto &match : *mAllMatches)
	{
		//Create error functor
		mErrorFunctors.emplace_back(new EpipolarSegmentErrorForPose(*match.sourceMeasurement, match.measurement, (float)FLAGS_MinDepth));
	}
}

std::vector<EssentialRansacModel> EssentialRansac::modelFromMinimalSet(const std::vector<int> &constraintIndices, const std::vector<int> &constraintSubindices)
{
	ProfileSection ss("essential::modelFrom5point");
	assert(constraintIndices.size()==6);

	std::vector<cv::Point2f> refp(5);
	std::vector<cv::Point2f> imgp(5);
	for(int i=0; i<5; ++i)
	{
		const int idx = constraintIndices[i];
		const int sidx = constraintSubindices[i];

		//XXX this will break with fish-eye lenses, but cv::findEssentialMat() can only take 2D points
		refp[i] = mRefXnPointsNormalized[idx];
		imgp[i] = mImageXnPointsNormalized[idx][sidx];
	}

	auto &extraMatch = mReferenceFrameMatches->at(constraintIndices[5]);

	//One extra match to validate the model
	//EssentialReprojectionError2D extraErrorFunctor(&extraMatch.sourceMeasurement->getCamera(), extraMatch.sourceMeasurement->getUniquePositionXn(), extraMatch.measurement.getOctave(), mCamera, extraMatch.measurement.getPositionXns());
	EpipolarSegmentErrorForPose extraErrorFunctor(*extraMatch.sourceMeasurement, extraMatch.measurement, (float)FLAGS_MinDepth);

	//Find essential: five point algorithm
	cv::Mat1f Eall ;
	{
	ProfileSection ss("essential::findEssential");
	Eall = cv::findEssentialMat(refp, imgp, 1, cv::Point2d(0,0));
	}
	const int modelCount = Eall.rows / 3;

	cv::Matx34f P1 = cv::Matx34f::eye();

	std::vector<EssentialRansacModel> models;
	for(int i=0; i<modelCount; ++i)
	{
		cv::Matx33f essential = Eall.rowRange(i*3,i*3+3);
		if(std::isnan(essential(0,0)))
		{
			//Replace with identity
			essential = cv::Matx33f(0,0,0, 0,0,-1, 0,1,0);
		}

		//Decompose essential
		cv::Matx33f R[2];
		cv::Vec3f t;
		cv::decomposeEssentialMat(essential, R[0], R[1], t);

		for(int kr=0; kr<2; ++kr)
		{
			for(int kt=0; kt<2; ++kt)
			{
				cv::Vec3f tk;
				if(kt==0)
					tk = t;
				else
					tk = -t;

				bool isGoodModel = true;

				////////////////////////////////////////////
				//Ensure a proper essential matrix
				// It turns out that opencv's 5 point algorithm might return
				// a matrix that is not a valid essential matrix (i.e. full rank).
				// Also, rebuild it to get correct sign (very important)
				const cv::Matx33f essentialk = EssentialUtils::EssentialFromPose(R[kr], tk);

				//Traingulate the 5 points used
				cv::Matx34f P2 = cvutils::CatRT(R[kr], tk);;
				cv::Mat1f p4mat;
				cv::triangulatePoints(P1,P2,refp,imgp,p4mat);

				//Make sure all 5 points are in front
				for(int j=0; j<5; ++j)
				{
					float z = p4mat(2,j);
					float w = p4mat(3,j);
					if(std::signbit(z) != std::signbit(w))
					{
						isGoodModel = false;
						break;
					}
				}

				//Change the reference frame of the pose
				cv::Matx33f worldR;
				cv::Vec3f worldT;

				//Eval the 6th point
				if (isGoodModel)
				{
					worldR = R[kr] * mReferenceFramePose.getRotationRef();
					worldT = R[kr] * mReferenceFramePose.getTranslationRef() + tk;

					std::vector<float> residuals;
					extraErrorFunctor.computeAllResiduals(worldR, worldT, residuals);

					MatchReprojectionErrors errors;
					CeresUtils::ResidualsToErrors<EpipolarSegmentError::kResidualsPerItem>(extraErrorFunctor.getPointCount(), residuals, mOutlierErrorThresholdSq, errors);

					if(!errors.isInlier)
					{
						//DTSLAM_LOG << "Model failed because of extra point\n";
						isGoodModel = false;
					}
				}

				if(isGoodModel)
				{
					//Model is good
					//Add model
					models.emplace_back(essentialk, FullPose3D(worldR,worldT));
				}
			}
		}
	}
	//DTSLAM_LOG << "Model count: " << models.size() << "\n";
    return std::move(models);
}

void EssentialRansac::getInliers(const EssentialRansacModel &model, int &inlierCount, float &errorSumSq, EssentialRansacData &data)
{
	ProfileSection ss("essential::getInliers");
	ceres::CauchyLoss robustLoss(mOutlierErrorThreshold);

	errorSumSq = 0;
	inlierCount = 0;
	data.reprojectionErrors.resize(mErrorFunctors.size());
	for(int i=0, end=mErrorFunctors.size(); i!=end; ++i)
	{
		auto &errorFunctor = *mErrorFunctors[i];
		auto &errors = data.reprojectionErrors[i];

		//errorFunctor.evalToErrors(model.pose.getRotationRef(), model.pose.getTranslationRef(), mOutlierErrorThreshold, errors);
		std::vector<float> residuals;
		errorFunctor.computeAllResiduals(model.pose.getRotationRef(), model.pose.getTranslationRef(), residuals);

		CeresUtils::ResidualsToErrors<EpipolarSegmentError::kResidualsPerItem>(errorFunctor.getPointCount(), residuals, mOutlierErrorThresholdSq, errors);

		if(errors.isInlier)
			inlierCount++;

		double robustError[3];
		robustLoss.Evaluate(errors.bestReprojectionErrorSq, robustError);
		errorSumSq += (float)robustError[0];
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EssentialUtils
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
cv::Point3f EssentialUtils::EpipoleFromEssential(const cv::Matx33f &E)
{

	cv::Vec3f Ew;
	cv::Matx33f Eu,Evt;
	cv::SVD::compute(E, Ew, Eu, Evt);

	return cv::Point3f(Eu(0,2),Eu(1,2),Eu(2,2));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// refineEssential
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool EssentialRefiner::getReprojectionErrors(const FeatureMatch &match,
		const cv::Vec3d &rparams,
		const cv::Vec3d &translation,
		MatchReprojectionErrors &errors)
{
	EpipolarSegmentErrorForPose err(*match.sourceMeasurement, match.measurement, (float)FLAGS_MinDepth);

	std::vector<double> residuals(EpipolarSegmentErrorForPose::kResidualsPerItem*err.getPointCount());
	err.computeAllResiduals(rparams.val, translation.val, residuals.data());
	CeresUtils::ResidualsToErrors<EpipolarSegmentErrorForPose::kResidualsPerItem>(err.getPointCount(), residuals, mOutlierPixelThresholdSq, errors);
	return errors.isInlier;
}

void EssentialRefiner::getInliers(const std::vector<FeatureMatch> &matches,
		const cv::Vec3d &rparams,
		const cv::Vec3d &translation,
		int &inlierCount,
		std::vector<MatchReprojectionErrors> &errors)
{
	inlierCount = 0;

	const int matchCount=matches.size();
	errors.resize(matchCount);
	for(int i=0; i<matchCount; i++)
	{
		if(getReprojectionErrors(matches[i], rparams, translation, errors[i]))
			inlierCount++;
	}
}

void EssentialRefiner::refineEssential(const std::vector<FeatureMatch> &matches,
		cv::Matx33f &rotation, cv::Vec3f &translation, int &inlierCount, std::vector<MatchReprojectionErrors> &errors)
{
	const int matchCount=matches.size();

	//Convert input to double
	const cv::Matx33d rotation_d = rotation;
	cv::Vec3d rparams_d;
	ceres::RotationMatrixToAngleAxis(ceres::RowMajorAdapter3x3(rotation_d.val), rparams_d.val);

	cv::Vec3d translation_d = translation;

	//Check inliers before
	int inlierCountBefore;
	getInliers(matches, rparams_d, translation_d, inlierCountBefore, errors);

	//Non-linear minimization with distortion
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.minimizer_progress_to_stdout = false;
	//options.function_tolerance = mOutlierPixelThreshold/100;
	//options.parameter_tolerance = 1e-5;
        
        // in some environments ceres uses std::tr1::shared_ptr, in others
        // it uses std::shared_ptr. Let's keep it simple by not using
        // make_shared.
	options.linear_solver_ordering.reset(new ceres::ParameterBlockOrdering());

	ceres::Problem problem;

	problem.AddParameterBlock(rparams_d.val, 3);
	problem.AddParameterBlock(translation_d.val, 3, new Fixed3DNormParametrization(1.0));
	problem.AddParameterBlock(translation_d.val, 3);
	options.linear_solver_ordering->AddElementToGroup(rparams_d.val, 1);
	options.linear_solver_ordering->AddElementToGroup(translation_d.val, 1);

	ceres::CauchyLoss robustLoss(mOutlierPixelThreshold);

	for(int i=0; i<matchCount; i++)
	{
		auto &match = matches[i];
		auto &refM = *match.measurement.getFeature().getMeasurements()[0];
		
		//		bool isInlier = inlierMask[i]!=-1;
//		if(!isInlier && mRefineOnlyInliers)
//			continue;

		const int scale = 1<<match.measurement.getOctave();
		//const int scale = 1;

		ceres::LossFunction *lossFunc_i = &robustLoss;
		ceres::ScaledLoss *scaledLoss = new ceres::ScaledLoss(lossFunc_i, scale, ceres::DO_NOT_TAKE_OWNERSHIP);

		//problem.AddResidualBlock(
		//		new ceres::AutoDiffCostFunction<PoseReprojectionError2D,2,3,3>(
		//				new PoseReprojectionError2D(match)),
		//				scaledLoss, rparams_d.val, translation_d.val);
		problem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<EpipolarSegmentErrorForPose, 2, 3, 3>(
			new EpipolarSegmentErrorForPose(refM, match.measurement, (float)FLAGS_MinDepth)),
			scaledLoss, rparams_d.val, translation_d.val);
	}

	ceres::Solver::Summary summary;

	ceres::Solve(options, &problem, &summary);

	//DTSLAM_LOG << summary.FullReport();

	//Extract result
	cv::Matx33d finalRotation_d;
	ceres::AngleAxisToRotationMatrix(rparams_d.val, ceres::RowMajorAdapter3x3(finalRotation_d.val));
	rotation = finalRotation_d;

	translation = translation_d;

	getInliers(matches, rparams_d, translation_d, inlierCount, errors);

	const int inlierPercentage = (int)(100*inlierCount / (float)matches.size());
	DTSLAM_LOG << "Essential estimation refinement: inliersBefore=" << inlierCountBefore << ", inliers=" << inlierCount << "/" << matches.size() <<  "(" << inlierPercentage << "%)\n";
}

} /* namespace dtslam */
