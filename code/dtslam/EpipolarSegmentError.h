/*
 * EpipolarSegmentError.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef EPIPOLARSEGMENTERROR_H_
#define EPIPOLARSEGMENTERROR_H_

#include <vector>
#include <opencv2/core.hpp>
#include "CameraModel.h"
#include "Pose3D.h"

namespace dtslam
{

/////////////////////////////////////////////////////////////////////////////////////////////////
//EpipolarSegmentError
/////////////////////////////////////////////////////////////////////////////////////////////////

class EpipolarSegmentError
{
public:
	static const int kResidualsPerItem = 2;

	EpipolarSegmentError(const SlamFeatureMeasurement &refM, const SlamFeatureMeasurement &imgM, float minDepth) :
		EpipolarSegmentError(refM.getUniquePositionXn(), minDepth, &imgM.getCamera(), &imgM.getPositions(), &imgM.getPositionXns(), imgM.getOctave())
	{
	}

	EpipolarSegmentError(const cv::Point3f &refXn, float minDepth, const CameraModel *imgCamera, const std::vector<cv::Point2f> *imgPoints, const std::vector<cv::Point3f> *imgXns, int octave) :
		mRefXn(refXn), mRefMinDepthX(minDepth*refXn), mImgCamera(imgCamera), mImgXns(imgXns), mScale((float)(1<<octave)), mPointCount(mImgXns->size())
	{
		mUJac.resize(mPointCount);
		mVJac.resize(mPointCount);

		for(int i=0; i!=mPointCount; ++i)
		{
			//mImgCamera->projectFromWorldJacobian(imgXns->at(i), mUJac[i], mVJac[i]);
			mImgCamera->projectFromWorldJacobianLUT(imgPoints->at(i), mUJac[i], mVJac[i]);
			mUJac[i] *= 1.0f / mScale;
			mVJac[i] *= 1.0f / mScale;
		}
	}

	int getPointCount() const { return mPointCount; }

	template<class T>
    void computeAllResiduals(const T * const refR, const T * const refT, const T * const imgR, const T * const imgT, T *allResiduals) const;

protected:
	const cv::Point3f mRefXn;
	const cv::Point3f mRefMinDepthX;
	const CameraModel *mImgCamera;
	const std::vector<cv::Point3f> *mImgXns;
	const float mScale;

	const int mPointCount;
	std::vector<cv::Vec3f> mUJac;
	std::vector<cv::Vec3f> mVJac;
};


/////////////////////////////////////////////////////////////////////////////////////////////////
//EpipolarSegmentErrorForPose
/////////////////////////////////////////////////////////////////////////////////////////////////

class EpipolarSegmentErrorForPose : public EpipolarSegmentError
{
public:
	EpipolarSegmentErrorForPose(const SlamFeatureMeasurement &refM, const SlamFeatureMeasurement &imgM, float minDepth):
		EpipolarSegmentError(refM, imgM, minDepth), mRefR(refM.getFramePose().getRotation()), mRefT(refM.getFramePose().getTranslation())
	{
	}

	template<class T>
    void computeAllResiduals(const T * const imgRparams, const T * const imgT, T *allResiduals) const;

	void computeAllResiduals(const cv::Matx33f &imgR, const cv::Vec3f &imgT, std::vector<float> &allResiduals) const;

	template<class T>
    bool operator()(const T * const imgRparams, const T * const imgT, T * residuals) const;

protected:
	const cv::Matx33f mRefR;
	const cv::Vec3f mRefT;
};


/////////////////////////////////////////////////////////////////////////////////////////////////
//EpipolarSegmentErrorForBA
/////////////////////////////////////////////////////////////////////////////////////////////////

class EpipolarSegmentErrorForBA : public EpipolarSegmentError
{
public:
	EpipolarSegmentErrorForBA(const SlamFeatureMeasurement &refM, const SlamFeatureMeasurement &imgM, float minDepth) :
		EpipolarSegmentError(refM, imgM, minDepth)
	{
	}

	template<class T>
	void computeAllResiduals(const T * const refRparams, const T * const refT, const T * const imgRparams, const T * const imgT, T *allResiduals) const;

	template<class T>
	bool operator()(const T * const refRparams, const T * const refT, const T * const imgRparams, const T * const imgT, T * residuals) const;
};

} /* namespace dtslam */

#include "EpipolarSegmentErrorImpl.hpp"

#endif /* EPIPOLARSEGMENTERROR_H_ */
