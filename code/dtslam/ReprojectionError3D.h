/*
 * PoseReprojectionError3D.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef POSEREPROJECTIONERROR3D_H_
#define POSEREPROJECTIONERROR3D_H_

#include <vector>
#include <opencv2/core.hpp>
#include "CameraModel.h"
#include "CeresUtils.h"
#include "PoseEstimationCommon.h"
#include "SlamMap.h"
#include "FeatureMatcher.h"

namespace dtslam
{

class ReprojectionError3D
{
public:
	ReprojectionError3D(const SlamFeatureMeasurement &m):
		ReprojectionError3D(&m.getKeyFrame().getCameraModel(), m.getOctave(), m.getPositions())
	{
	}

	ReprojectionError3D(const CameraModel * const camera, const int octave, const std::vector<cv::Point2f> &imgPoints):
		mCamera(camera), mScale(1<<octave), mImgPoints(imgPoints), mImagePointCount(imgPoints.size())
	{
	}

	int getImagePointCount() const {return mImagePointCount;}

	template<class T>
    void computeAllResidualsRparams(const T * const rparams, const T * const t, const T * const x, T *allResiduals) const;

	template<class T>
    void computeAllResidualsRmat(const T * const R, const T * const t, const T * const x, T *allResiduals) const;

	template<class T>
	void residualsToErrors(const std::vector<T> &allResiduals, const float errorThreshold, MatchReprojectionErrors &errors) const;

	void evalToErrors(const cv::Matx33f &R, const cv::Vec3f &t, const cv::Vec3f &x, const float errorThreshold, MatchReprojectionErrors &errors) const;
	void evalToErrors(const cv::Vec3d &Rparams, const cv::Vec3d &t, const cv::Vec3d &x, const float errorThreshold, MatchReprojectionErrors &errors) const;

protected:
	const CameraModel * const mCamera;
	const int mScale;
	const std::vector<cv::Point2f> mImgPoints; //Note: this cannot be a reference because SlamFeatureMeasurements only have one position.
	const int mImagePointCount;

	template<class T>
    void pointResiduals(const T &u, const T &v, const int i, T *residuals) const;

	template<class T>
    void computeAllResidualsXc(const T * const xc, T *allResiduals) const;
};

class PoseReprojectionError3D: public ReprojectionError3D
{
public:
	PoseReprojectionError3D(const FeatureMatch &match):
		PoseReprojectionError3D(&match.measurement.getCamera(), match.measurement.getFeature().getPosition(), match.measurement.getOctave(), match.measurement.getPositions())
	{
	}

	PoseReprojectionError3D(const CameraModel * const camera, const cv::Point3f &featurePosition, const int octave, const std::vector<cv::Point2f> &imgPoints):
		ReprojectionError3D(camera, octave, imgPoints), mFeaturePosition(featurePosition)
	{
	}

	const cv::Vec3f &getFeaturePosition() const {return mFeaturePosition;}

	template<class T>
    bool operator()(const T * const rparams, const T * const t, T *residuals) const;

	void evalToErrors(const cv::Matx33f &R, const cv::Vec3f &t, const float errorThreshold, MatchReprojectionErrors &errors) const
	{
		ReprojectionError3D::evalToErrors(R,t,mFeaturePosition,errorThreshold,errors);
	}

	void evalToErrors(const cv::Vec3d &Rparams, const cv::Vec3d &t, const float errorThreshold, MatchReprojectionErrors &errors) const
	{
		ReprojectionError3D::evalToErrors(Rparams,t,mFeaturePosition,errorThreshold,errors);
	}

protected:
	const cv::Vec3f mFeaturePosition;
};

class BAReprojectionError3D: public ReprojectionError3D
{
public:
	BAReprojectionError3D(const CameraModel * const camera, const int octave, const std::vector<cv::Point2f> &imgPoints):
		ReprojectionError3D(camera, octave, imgPoints)
	{
	}

	BAReprojectionError3D(const SlamFeatureMeasurement &measurement):
		BAReprojectionError3D(&measurement.getKeyFrame().getCameraModel(), measurement.getOctave(), measurement.getPositions())
	{
	}

	template<class T>
    bool operator()(const T * const rparams, const T * const t, const T * const x, T *residuals) const;

protected:
};

}

//Include implementation of templates
#include "ReprojectionError3DImpl.hpp"

#endif /* POSEREPROJECTIONERROR3D_H_ */
