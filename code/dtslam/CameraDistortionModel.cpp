/*
 * CameraDistortionModel.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "CameraDistortionModel.h"
#include <opencv2/imgproc.hpp>
#include <ceres/ceres.h>
#include "cvutils.h"

namespace dtslam {

cv::FileStorage& operator << (cv::FileStorage& fs, const NullCameraDistortionModel &dist)
{
	return fs << "null";
}

void operator >> (const cv::FileNode& root, NullCameraDistortionModel& dist)
{
}

cv::FileStorage& operator << (cv::FileStorage& fs, const RadialCameraDistortionModel &dist)
{
	fs << "{";
	fs << "k1" << dist.mK1;
	fs << "k2" << dist.mK2;
	fs << "maxRadiusSq" << dist.mMaxRadiusSq;
	fs << "}";
	return fs;
}

void operator >> (const cv::FileNode& root, RadialCameraDistortionModel& dist)
{
	float k1,k2,maxRadiusSq;
	root["k1"] >> k1;
	root["k2"] >> k2;
	root["maxRadiusSq"] >> maxRadiusSq;
	dist.init(k1,k2);
	dist.setMaxRadius(maxRadiusSq);
}

//This class is for using ceres for the undistortion. Too slow to use ceres though.
//class RadialCameraUndistortError
//{
//public:
//	RadialCameraUndistortError(const RadialCameraDistortionModel *model, double xd, double yd):
//		mModel(model), mXd(xd), mYd(yd)
//	{
//	}
//
//	template<class T>
//    bool operator()(const T * const rfactor, T *residuals) const
//	{
//		T xn = T(mXd)*rfactor[0];
//		T yn = T(mYd)*rfactor[0];
//		T xdd;
//		T ydd;
//
//		mModel->distortPoint(xn,yn,xdd,ydd);
//		residuals[0] = xdd-T(mXd);
//		residuals[1] = ydd-T(mYd);
//		return true;
//	}
//
//private:
//	const RadialCameraDistortionModel *mModel;
//	double mXd;
//	double mYd;
//
//};

cv::Point3f RadialCameraDistortionModel::undistortPoint(const cv::Point2f &pd) const
{
	//This code uses opencv functions:
//	std::vector<cv::Point2f> pdv(1, pd);
//	std::vector<cv::Point2f> pnv(1);
//	cv::undistortPoints(pdv,pnv,cv::Matx33f(1,0,0,0,1,0,0,0,1), cv::Matx14f(mK1, mK2, 0, 0));
//	return pnv[0];

	//This code is a copy paste from opencv library, it is much faster because of the input parameter conversion.
//	const int kMaxIters = 11;
//	double x = pd.x;
//	double x0 = pd.x;
//	double y = pd.y;
//	double y0 = pd.y;
//	double k[12]={0,0,0,0,0,0,0,0,0,0,0};
//	k[0] = mK1;
//	k[1] = mK2;
//    for( int j = 0; j < kMaxIters; j++ )
//    {
//        double r2 = x*x + y*y;
//        double icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
//        double deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2;
//        double deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2;
//        x = (x0 - deltaX)*icdist;
//        y = (y0 - deltaY)*icdist;
//    }

		const int kMaxIters = 11;
		float x = pd.x;
		float y = pd.y;
	    for( int j = 0; j < kMaxIters; j++ )
	    {
	    	float r2 = x*x + y*y;
	    	float icdist = 1+r2*mK1+r2*r2*mK2;
	        x = (pd.x)/icdist;
	        y = (pd.y)/icdist;
	    }

	return cvutils::PointToHomogenousUnitNorm(cv::Point2f(x,y));

//    //This code uses ceres (waaaaaay too slow)
//	double rfactor=1.0;
//
//	ceres::Problem problem;
//
//	problem.AddResidualBlock(
//			new ceres::AutoDiffCostFunction<RadialCameraUndistortError,2,1>(
//					new RadialCameraUndistortError(this, pd.x, pd.y)),
//			NULL, &rfactor);
//
//	ceres::Solver::Options options;
//	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
//	options.minimizer_progress_to_stdout = false;
//
//	ceres::Solver::Summary summary;
//
//	ceres::Solve(options, &problem, &summary);
//
//	return pd*rfactor;
}

} /* namespace dtslam */
