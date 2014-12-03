/*
 * CameraModel.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef CAMERAMODEL_H_
#define CAMERAMODEL_H_

#include <memory>
#include <opencv2/core.hpp>
#include <gflags/gflags.h>
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <ceres/rotation.h>
#include "CameraDistortionModel.h"
#include "log.h"
#include "Serializer.h"

namespace dtslam
{

//Forward
template<class TDistortionModel>
class CameraModel_;

//This typedef decides which camera model to use in the system
//It has to be a typedef because ceres requires templated functions
//to build the automatic diff cost functions and templated methods cannot be virtual.
typedef CameraModel_<RadialCameraDistortionModel> CameraModel;

template<class TDistortionModel>
class CameraModel_: public ISerializable
{
public:
	CameraModel_()
	{
	}

	void init(float fx, float fy, float u0, float v0, int width, int height)
	{
		mFx = fx;
		mFy = fy;
		mU0 = u0;
		mV0 = v0;
		mImageSize = cv::Size(width, height);
	}

	template<class T>
	friend cv::FileStorage& operator << (cv::FileStorage& fs, const CameraModel_<T> &camera);
	template<class T>
	friend void operator >> (const cv::FileNode& root, CameraModel_<T>& camera);

	float getFx() const {return mFx;}
	float getFy() const {return mFy;}
	float getU0() const {return mU0;}
	float getV0() const {return mV0;}
	const cv::Size &getImageSize() const {return mImageSize;}
	TDistortionModel &getDistortionModel() {return mDistortionModel;}
	const TDistortionModel &getDistortionModel() const {return mDistortionModel;}

	float getMaxRadiusSq(const cv::Size2i &imageSize) const;

	cv::Matx33f getK() const {return cv::Matx33f(mFx,0,mU0, 0,mFy,mV0, 0,0,1);}

	void initLUT();

	bool isPointInside(const cv::Point3f &xc, const cv::Point2f &p) const
	{
		return xc.z > 0 &&  //Depth is positive
			p.x >= 0 && p.y >= 0 && p.x < mImageSize.width && p.y < mImageSize.height; //Inside image
	}

	//////////////////////////////////////
	// Project form world
	// Same code, one using return value, one using args by-reference (for ceres)
	cv::Point2f projectFromWorld(const cv::Point3f &xc) const
	{
		return projectFromDistorted(mDistortionModel.distortPoint(xc));
	}
	template<class T>
	void projectFromWorld(const T &x, const T &y, const T &z, T &u, T&v) const
	{
		T xd, yd;
		mDistortionModel.distortPoint(x,y,z,xd,yd);
		projectFromDistorted(xd,yd,u,v);
	}

	void projectFromWorldJacobian(const cv::Point3f &xc, cv::Vec3f &ujac, cv::Vec3f &vjac) const;
	void projectFromWorldJacobianLUT(const cv::Point2i &uv, cv::Vec3f &ujac, cv::Vec3f &vjac) const
	{
		assert(uv.x >= 0 && uv.x < mProjectFromWorldJacobianLUT.cols && uv.y >= 0 && uv.y < mProjectFromWorldJacobianLUT.rows);
		const cv::Vec6f &lut = mProjectFromWorldJacobianLUT(uv.y, uv.x);
		ujac[0] = lut[0];
		ujac[1] = lut[1];
		ujac[2] = lut[2];
		vjac[0] = lut[3];
		vjac[1] = lut[4];
		vjac[2] = lut[5];
	}

	//////////////////////////////////////
	// Unproject (assuming z=1)
	// Note that often no analytical formula is present, therefore we have no templated version for ceres
	// Also, use integer pixel positions so that we can use a LUT for undistortion
	cv::Point3f unprojectToWorld(const cv::Point2f &uv) const
	{
		return mDistortionModel.undistortPoint(unprojectToDistorted(uv));
	}
	//cv::Point3f unprojectToWorldLUT(const cv::Point2f &uv) const
	//{
	//	return unprojectToWorld(uv);
	//}
	cv::Point3f unprojectToWorldLUT(const cv::Point2i &uv) const
	{
		//if (!(uv.x >= 0 && uv.x < mUnprojectLUT.cols && uv.y >= 0 && uv.y < mUnprojectLUT.rows))
		//{
			//DTSLAM_LOG << "Here!!!!!!!!!!!!!!!!!!!!!!!!!\n";
			assert(uv.x >= 0 && uv.x < mUnprojectLUT.cols && uv.y >= 0 && uv.y < mUnprojectLUT.rows);
		//}
		return mUnprojectLUT(uv.y, uv.x);
	}

	/////////////////////////////////////
	// Serialization
	static const std::string GetTypeName() { return "Camera"; }
	const std::string getTypeName() const { return GetTypeName(); }
	void serialize(Serializer &s, cv::FileStorage &fs) const;
	void deserialize(Deserializer &s, const cv::FileNode &node);

protected:
	//Four parameters of the intrinsic matrix
	float mFx;
	float mFy;
	float mU0;
	float mV0;
	cv::Size2i mImageSize;

	//Distortion model
	TDistortionModel mDistortionModel;

	cv::Mat3f mUnprojectLUT; //Unproject doesn't need an analytical formula. If it is slow we can cache it in a LUT.
	cv::Mat_<cv::Vec6f> mProjectFromWorldJacobianLUT; //Can be cached if the pixel value is known

	//////////////////////////////////////
	// Project form distorted
	// Same code, one using return value, one using args by-reference (for ceres)
	cv::Point2f projectFromDistorted(const cv::Point2f &pd) const
	{
		return cv::Point2f(mFx*pd.x + mU0, mFy*pd.y + mV0);
	}
	template<class T>
	void projectFromDistorted(const T &xd, const T &yd, T &u, T &v) const
	{
		u = T(mFx)*xd + T(mU0);
		v = T(mFy)*yd + T(mV0);
	}

	cv::Point2f unprojectToDistorted(const cv::Point2f &uv) const
	{
		return cv::Point2f((uv.x - mU0) / mFx, (uv.y - mV0) / mFy);
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Template implementations
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class TDistortionModel>
cv::FileStorage& operator << (cv::FileStorage& fs, const CameraModel_<TDistortionModel> &camera)
{
	fs << "{";
	fs << "fx" << camera.mFx;
	fs << "fy" << camera.mFy;
	fs << "u0" << camera.mU0;
	fs << "v0" << camera.mV0;
	fs << "width" << camera.mImageSize.width;
	fs << "height" << camera.mImageSize.height;
	fs << "distortion" << camera.mDistortionModel;
	fs << "}";
	return fs;
}

template<class TDistortionModel>
void operator >> (const cv::FileNode& root, CameraModel_<TDistortionModel>& camera)
{
	float fx,fy,u0,v0;
	int w, h;
    root["fx"] >> fx;
    root["fy"] >> fy;
    root["u0"] >> u0;
    root["v0"] >> v0;
    root["width"] >> w;
    root["height"] >> h;
    camera.init(fx,fy,u0,v0,w,h);
    root["distortion"] >> camera.mDistortionModel;
	camera.initLUT();
}

template<class TDistortionModel>
float CameraModel_<TDistortionModel>::getMaxRadiusSq(const cv::Size2i &imageSize) const
{
	cv::Point2i corners[] = {cv::Point2i(0,0), cv::Point2i(0,imageSize.height),
			cv::Point2i(imageSize.width,imageSize.height), cv::Point2i(imageSize.width,0)};

	float maxRadiusSq = 0;
	for(int i=0; i<4; ++i)
	{
		const cv::Point2f xn = cvutils::NormalizePoint(unprojectToWorld(corners[i]));
		const float r2 = xn.x*xn.x + xn.y*xn.y;
		if(r2>maxRadiusSq)
			maxRadiusSq = r2;
	}

	return maxRadiusSq;
}

template<class TDistortionModel>
void CameraModel_<TDistortionModel>::initLUT()
{
	DTSLAM_LOG << "Building LUTs for image of size " << mImageSize << "...";
	mUnprojectLUT.create(mImageSize);
	mProjectFromWorldJacobianLUT.create(mImageSize);
	for(int v=0; v<mImageSize.height; v++)
	{
		cv::Vec3f *rowUnprojectLUT = mUnprojectLUT[v];
		cv::Vec6f *rowProjectLUT = mProjectFromWorldJacobianLUT[v];
		for (int u = 0; u<mImageSize.width; u++)
		{
			rowUnprojectLUT[u] = unprojectToWorld(cv::Point2i(u,v));

			cv::Vec3f ujac, vjac;
			projectFromWorldJacobian(rowUnprojectLUT[u], ujac, vjac);
			rowProjectLUT[u] = cv::Vec6f(ujac[0], ujac[1], ujac[2], vjac[0], vjac[1], vjac[2]);
		}
	}
	DTSLAM_LOG << "done\n";
}

template<class TDistortionModel>
void CameraModel_<TDistortionModel>::serialize(Serializer &s, cv::FileStorage &fs) const
{
	fs << "fx" << mFx;
	fs << "fy" << mFy;
	fs << "u0" << mU0;
	fs << "v0" << mV0;
	fs << "width" << mImageSize.width;
	fs << "height" << mImageSize.height;
	fs << "distortion" << mDistortionModel;
}

template<class TDistortionModel>
void CameraModel_<TDistortionModel>::deserialize(Deserializer &s, const cv::FileNode &node)
{
	node >> *this;
}

} /* namespace dtslam */

#endif /* CAMERAMODEL_H_ */
