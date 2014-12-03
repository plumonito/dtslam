/*
 * EssentialUtils.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef ESSENTIALUTILS_H_
#define ESSENTIALUTILS_H_

#include <opencv2/core.hpp>
#include "Pose3D.h"
#include "BaseRansac.h"
#include "CameraModel.h"
#include "SlamMap.h"
#include "cvutils.h"
#include "FeatureMatcher.h"
#include "CeresUtils.h"

namespace dtslam
{

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EssentialUtils
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class EssentialUtils
{
public:
	inline static cv::Matx33f EssentialFromPose(const Pose3D &ref, const Pose3D &img);
	inline static cv::Matx33f EssentialFromPose(const Pose3D &relative);
	inline static cv::Matx33f EssentialFromPose(const cv::Matx33f &relativeR, const cv::Vec3f &relativeT);
	static cv::Point3f EpipoleFromEssential(const cv::Matx33f &E);

	template<class T>
	inline static void EssentialFromPose(const T * const relativeR, const T * const relativeT, T * essential);
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementations

//Equation: imgPoint * Essential * refPoint = 0
cv::Matx33f EssentialUtils::EssentialFromPose(const Pose3D &ref, const Pose3D &img)
{
	FullPose3D relative = FullPose3D::MakeRelativePose(ref, img);
	return EssentialFromPose(relative.getRotationRef(), relative.getTranslationRef());
}
cv::Matx33f EssentialUtils::EssentialFromPose(const Pose3D &relative)
{
	return EssentialFromPose(relative.getRotation(), relative.getTranslation());
}
cv::Matx33f EssentialUtils::EssentialFromPose(const cv::Matx33f &relativeR, const cv::Vec3f &relativeT)
{
	return cvutils::SkewSymmetric(relativeT)*relativeR;
}

template<class T>
void EssentialUtils::EssentialFromPose(const T * const relativeR, const T * const relativeT, T * essential)
{
	auto relativeRMat = CeresUtils::FixedRowMajorAdapter3x3(relativeR);
	auto essentialMat = CeresUtils::FixedRowMajorAdapter3x3(essential);

	//Skew symmetric is (0,-t[2],t[1],  t[2],0,-t[0],  -t[1],t[0],0);
	essentialMat(0,0) = relativeT[1]*relativeRMat(2,0) - relativeT[2]*relativeRMat(1,0);
	essentialMat(0,1) = relativeT[1]*relativeRMat(2,1) - relativeT[2]*relativeRMat(1,1);
	essentialMat(0,2) = relativeT[1]*relativeRMat(2,2) - relativeT[2]*relativeRMat(1,2);

	essentialMat(1,0) = relativeT[2]*relativeRMat(0,0) - relativeT[0]*relativeRMat(2,0);
	essentialMat(1,1) = relativeT[2]*relativeRMat(0,1) - relativeT[0]*relativeRMat(2,1);
	essentialMat(1,2) = relativeT[2]*relativeRMat(0,2) - relativeT[0]*relativeRMat(2,2);

	essentialMat(2,0) = relativeT[0]*relativeRMat(1,0) - relativeT[1]*relativeRMat(0,0);
	essentialMat(2,1) = relativeT[0]*relativeRMat(1,1) - relativeT[1]*relativeRMat(0,1);
	essentialMat(2,2) = relativeT[0]*relativeRMat(1,2) - relativeT[1]*relativeRMat(0,2);
}

}

#endif /* ESSENTIALUTILS_H_ */
