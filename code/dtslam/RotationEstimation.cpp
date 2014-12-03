/*
 * RotationEstimator.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "RotationEstimation.h"
#include <cassert>

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/calib3d.hpp>

#include "cvutils.h"
#include "CeresUtils.h"

namespace dtslam
{
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Rotation3DRansac
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Rotation3DRansac::setData(const std::vector<FeatureMatch> *matches)
{
	assert(matches!=NULL);
	mMatches = matches;
	mConstraintCount = mMatches->size();

	mFeatureDirections.resize(mConstraintCount);
	for(int i=0; i<mConstraintCount; ++i)
	{
		auto &match = mMatches->at(i);
		mFeatureDirections[i] = match.sourceMeasurement->getFramePose().getRotation().t()*match.sourceMeasurement->getUniquePositionXn();
	}
}

std::vector<cv::Matx33f> Rotation3DRansac::modelFromMinimalSet(const std::vector<int> &constraintIndices, const std::vector<int> &subconstraintIndices)
{
	auto &match1 = mMatches->at(constraintIndices[0]);
	auto &match2 = mMatches->at(constraintIndices[1]);

	const cv::Point3f &X1 = mFeatureDirections[constraintIndices[0]];
	const cv::Point3f &X2 = mFeatureDirections[constraintIndices[1]];
	cv::Matx32f X(X1.x, X2.x, X1.y, X2.y, X1.z, X2.z);

	const cv::Point3f &Y1 = match1.measurement.getPositionXns()[subconstraintIndices[0]];
	const cv::Point3f &Y2 = match2.measurement.getPositionXns()[subconstraintIndices[1]];
	cv::Matx23f Yt(Y1.x, Y1.y, Y1.z, Y2.x, Y2.y, Y2.z);

	cv::Matx33f XYt = X*Yt;
	cv::SVD svd(XYt, cv::SVD::MODIFY_A);

    std::vector<cv::Matx33f> resVec(1);
	cv::Matx33f &R = resVec[0];
	R = cv::Mat(svd.vt.t() * svd.u.t());
    if (determinant(R) < 0)
    {
        R = -R;
    }

    return resVec;
}

//This function should be ok but it is not used for now
//
//cv::Matx33f Rotation3DRansac::modelFromInliers(const std::vector<int> &constraintIndices)
//{
//	cv::Mat1f X(3, constraintIndices.size());
//	cv::Mat1f Yt(constraintIndices.size(),3);
//	for(int i=0; i<(int)constraintIndices.size(); ++i)
//	{
//		const cv::Point3f &Xi = (*mFeatures)[constraintIndices[i]];
//		X(0,i) = Xi.x;
//		X(1,i) = Xi.y;
//		X(2,i) = Xi.z;
//
//		const cv::Point3f &Yi = (*mImageXnPoints)[constraintIndices[i]];
//		Yt(i,0) = Yi.x;
//		Yt(i,1) = Yi.y;
//		Yt(i,2) = Yi.z;
//	}
//
//	cv::Mat1f XYt = X*Yt;
//	cv::SVD svd(XYt, cv::SVD::MODIFY_A);
//
//	cv::Matx33f R;
//	R = cv::Mat(svd.vt.t() * svd.u.t());
//    if (determinant(R) < 0)
//    {
//        R = -R;
//    }
//    return R;
//}

void Rotation3DRansac::getInliers(const cv::Matx33f &model, int &inlierCount, float &errorSumSq, std::vector<int> &inliers)
{
	ceres::CauchyLoss robustLoss(mOutlierErrorThreshold);

	inlierCount = 0;
	errorSumSq = 0;
	inliers.resize(mConstraintCount);

	auto &camera = mMatches->at(0).sourceMeasurement->getCamera();

	for(int i=0; i<mConstraintCount; ++i)
	{
		auto &match = mMatches->at(i);
		const int scale = 1<<match.measurement.getOctave();
		const int scaleSq = scale*scale;
		const cv::Point3f &Xi = mFeatureDirections[i];
		const cv::Point3f imgXnp = model*Xi;

		float minError = mOutlierErrorThresholdSq;
		int minIdx = -1;
		for(int j=0, end=(int)match.measurement.getPositionXns().size(); j<end; ++j)
		{
			const cv::Point2f &imgUv = match.measurement.getPositions()[j];
			const cv::Point3f &imgXn = match.measurement.getPositionXns()[j];
			const cv::Point3f diffXn = imgXnp - imgXn;
			
			cv::Vec3f ujac, vjac;
			//camera.projectFromWorldJacobian(imgXn, ujac, vjac);
			camera.projectFromWorldJacobianLUT(imgUv,ujac,vjac);

			float du = ujac.dot(diffXn);
			float dv = vjac.dot(diffXn);
			const float error = (du*du+dv*dv) / (scaleSq);
			if(error < minError)
			{
				minError = error;
				minIdx = j;
			}
		}

		if(minError < mOutlierErrorThresholdSq)
		{
			inlierCount++;
			inliers[i] = minIdx;
		}
		else
			inliers[i] = -1;

		double robustError[3];
		robustLoss.Evaluate(minError, robustError);
		errorSumSq += (float)robustError[0];
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// RotationRefiner
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class RotationError
{
public:
	RotationError(const CameraModel * const camera, const cv::Point3f &featureDirection, const std::vector<cv::Point2f> &imagePoints, int octave):
		mCamera(camera), mFeatureDirection(featureDirection), mImagePoints(imagePoints), mImagePointCount(imagePoints.size()), mScale(1<<octave)
	{
	}

	template<class T>
    void pointResiduals(const T &u, const T &v, const int i, T *residuals) const
	{
		const cv::Point2f &p = mImagePoints[i];

		residuals[0] = (u-T(p.x))/T(mScale);
		residuals[1] = (v-T(p.y))/T(mScale);
	}

	template<class T>
    void computeAllResiduals(const T * const rparams, T *allResiduals) const
	{
		T xw[3] = {T(mFeatureDirection.x), T(mFeatureDirection.y), T(mFeatureDirection.z)};

		T xc[3];
		ceres::AngleAxisRotatePoint(rparams, xw, xc);

		T u,v;
		mCamera->projectFromWorld(xc[0],xc[1],xc[2],u,v);

		//Calculate residuals for all points
		for(int i=0; i<mImagePointCount; ++i)
		{
			pointResiduals(u,v,i,allResiduals+2*i);
		}
	}

	template<class T>
    bool operator()(const T * const rparams, T * residuals, int &minIndex) const
	{
		if(mImagePointCount==1)
		{
			computeAllResiduals(rparams, residuals);
			minIndex = 0;
		}
		else
		{
			std::vector<T> allResiduals(2*mImagePointCount);
			computeAllResiduals(rparams, allResiduals.data());

			double minSq = CeresUtils::NormSq<2>(allResiduals.data());
			minIndex = 0;
			residuals[0] = allResiduals[0];
			residuals[1] = allResiduals[1];

			for(int i=1; i!=mImagePointCount; ++i)
			{
				const double sq = CeresUtils::NormSq<2>(&allResiduals[2*i]);
				if(sq<minSq)
				{
					minSq = sq;
					minIndex = i;
					residuals[0] = allResiduals[2*i];
					residuals[1] = allResiduals[2*i+1];
				}
			}
		}
		return true;
	}

	template<class T>
    bool operator()(const T * const rparams, T *residuals) const
	{
		int minIndex;
		return (*this)(rparams, residuals, minIndex);
	}

protected:
	const CameraModel * const mCamera;
	const cv::Point3f mFeatureDirection;
	const std::vector<cv::Point2f> &mImagePoints;
	const int mImagePointCount;
	int mScale;
};

bool RotationRefiner::getReprojectionErrors(const FeatureMatch &match,
		const cv::Vec3d &rparams,
		MatchReprojectionErrors &errors)
{
	int pointCount = match.measurement.getPositionCount();
	std::vector<double> allResiduals(2*pointCount);

	cv::Point3f featureDirection = match.sourceMeasurement->getFramePose().getRotation().t()*match.sourceMeasurement->getUniquePositionXn();
	RotationError err(mCamera, featureDirection, match.measurement.getPositions(), match.measurement.getOctave());
	err.computeAllResiduals(rparams.val, allResiduals.data());
	CeresUtils::ResidualsToErrors<2>(pointCount, allResiduals, mOutlierPixelThresholdSq, errors);

	return errors.isInlier;
}

void RotationRefiner::getInliers(const std::vector<FeatureMatch> &matches,
		const cv::Vec3d &rparams,
		int &inlierCount,
		std::vector<MatchReprojectionErrors> &errors)
{
	inlierCount = 0;

	const int matchCount=matches.size();
	errors.resize(matchCount);

	for(int i=0; i<matchCount; i++)
	{
		if(getReprojectionErrors(matches[i], rparams, errors[i]))
			inlierCount++;
	}
}

void RotationRefiner::refineRotation(const std::vector<FeatureMatch> &matches,
		const cv::Vec3f &center,
		cv::Matx33f &rotation,
		int &inlierCount,
		std::vector<MatchReprojectionErrors> &errors)
{
	const int matchCount=matches.size();

	const cv::Matx33d rotation_d = rotation;
	cv::Vec3d rparams_d;
	ceres::RotationMatrixToAngleAxis(ceres::RowMajorAdapter3x3(rotation_d.val), rparams_d.val);

	//Check inliers before
	int inlierCountBefore;
	getInliers(matches, rparams_d, inlierCountBefore, errors);
	//DTSLAM_LOG << "Rotation refine: Inliers before=" << inlierCountBefore << "\n";

	//Non-linear minimization with distortion
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	//options.function_tolerance = mOutlierPixelThreshold/100;
	//options.parameter_tolerance = 1e-5;
	options.minimizer_progress_to_stdout = false;
	options.logging_type = ceres::SILENT;

	ceres::Problem problem;

	problem.AddParameterBlock(rparams_d.val, 3);

	ceres::CauchyLoss robustLoss(mOutlierPixelThreshold);

	for(int i=0; i<matchCount; i++)
	{
		auto &match = matches[i];
		const int scale = 1<<match.measurement.getOctave();

		ceres::LossFunction *lossFunc_i = &robustLoss;
		ceres::ScaledLoss *imgScaledLoss = new ceres::ScaledLoss(lossFunc_i, scale, ceres::DO_NOT_TAKE_OWNERSHIP);

		cv::Point3f featureDirection = match.sourceMeasurement->getFramePose().getRotation().t()*match.sourceMeasurement->getUniquePositionXn();
		problem.AddResidualBlock(
				new ceres::AutoDiffCostFunction<RotationError,2,3>(
						new RotationError(mCamera, featureDirection, match.measurement.getPositions(), match.measurement.getOctave())),
				imgScaledLoss, rparams_d.val);
	}

	ceres::Solver::Summary summary;

	ceres::Solve(options, &problem, &summary);

	//DTSLAM_LOG << summary.FullReport();

	//Extract result
	cv::Matx33d finalRotation_d;
	ceres::AngleAxisToRotationMatrix(rparams_d.val, ceres::RowMajorAdapter3x3(finalRotation_d.val));
	rotation = finalRotation_d;

	//Check inliers
	getInliers(matches, rparams_d, inlierCount, errors);
}

} /* namespace dtslam */
