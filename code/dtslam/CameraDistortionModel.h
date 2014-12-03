/*
 * CameraDistortionModel.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef CAMERADISTORTIONMODEL_H_
#define CAMERADISTORTIONMODEL_H_

#include <opencv2/core.hpp>
#include "cvutils.h"

namespace dtslam
{

class NullCameraDistortionModel
{
public:
	NullCameraDistortionModel() {}

	friend cv::FileStorage& operator << (cv::FileStorage& fs, const NullCameraDistortionModel &dist);
	friend void operator >> (const cv::FileNode& root, NullCameraDistortionModel& dist);

	cv::Point2f distortPoint(const cv::Point3f &x) const
	{
		return cvutils::NormalizePoint(x);
	}
	template<class T>
	void distortPoint(const T &x, const T &y, const T &z, T &xd, T &yd) const
	{
		xd=x/z;
		yd=y/z;
	}

	cv::Point3f undistortPoint(const cv::Point2f &pd) const
	{
		return cvutils::PointToHomogenousUnitNorm(pd);
	}
};

class RadialCameraDistortionModel
{
public:
	RadialCameraDistortionModel() {}

	void init(float k1, float k2)
	{
		mK1 = k1;
		mK2 = k2;
		mMaxRadiusSq = 1e10;
	}

	float getK1() const {return mK1;}
	float getK2() const {return mK2;}

	float getMaxRadiusSq() const {return mMaxRadiusSq;}
	void setMaxRadius(float maxRadiusSq)
	{
		mMaxRadiusSq = maxRadiusSq;
	}


	friend cv::FileStorage& operator << (cv::FileStorage& fs, const RadialCameraDistortionModel &dist);
	friend void operator >> (const cv::FileNode& root, RadialCameraDistortionModel& dist);

	cv::Point2f distortPoint(const cv::Point3f &x) const
	{
		const cv::Point2f xn = cvutils::NormalizePoint(x);
		float r2 = xn.x*xn.x + xn.y*xn.y;
		if(r2 > mMaxRadiusSq)
			r2 = mMaxRadiusSq;

		const float r4=r2*r2;
		const float factor = 1+mK1*r2+mK2*r4;
		return cv::Point2f(xn.x*factor, xn.y*factor);
	}
	template<class T>
	void distortPoint(const T &x, const T &y, const T &z, T &xd, T &yd) const
	{
		T xn = x/z;
		T yn = y/z;

		T r2 = xn*xn + yn*yn;
		T maxR2(mMaxRadiusSq);
		if(r2 > maxR2)
			r2 = maxR2;
		T r4=r2*r2;
		T factor = T(1)+T(mK1)*r2+T(mK2)*r4;
		xd = xn*factor;
		yd = yn*factor;
	}

	//Non-linear, uses ceres
	cv::Point3f undistortPoint(const cv::Point2f &pd) const;

protected:
	float mK1;
	float mK2;
	float mMaxRadiusSq; //Do not apply distortion after this radius. Keeps the distortion within the image limits where it was calibrated.
};
} /* namespace dtslam */

#endif /* CAMERADISTORTIONMODEL_H_ */
