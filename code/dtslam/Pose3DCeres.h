/*
* Pose3DCeres.h
*
* Copyright(C) 2014, University of Oulu, all rights reserved.
* Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
* Third party copyrights are property of their respective owners.
* Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
*          Kihwan Kim(kihwank@nvidia.com)
* Author : Daniel Herrera C.
*/
#ifndef POSE3DCERES_H_
#define POSE3DCERES_H_

#include "Pose3D.h"
#include "CeresUtils.h"

namespace dtslam
{
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Template implementations

template<class T>
void Pose3D::Apply3D(const Pose3D *pose, const T * const point, T*result)
{
	{
		const FullPose3D *realPose = dynamic_cast<const FullPose3D*>(pose);
		if(realPose)
		{
			realPose->apply3D(point, result);
			return;
		}
	}

	{
		const RelativeRotationPose3D *realPose = dynamic_cast<const RelativeRotationPose3D*>(pose);
		if(realPose)
		{
			realPose->apply3D(point, result);
			return;
		}
	}

	{
		const RelativePose3D *realPose = dynamic_cast<const RelativePose3D*>(pose);
		if(realPose)
		{
			realPose->apply3D(point, result);
			return;
		}
	}

	assert(false);
}

template<class T>
void Pose3D::Apply4D(const Pose3D *pose, const T * const point, T*result)
{
	{
		const FullPose3D *realPose = dynamic_cast<const FullPose3D*>(pose);
		if(realPose)
		{
			realPose->apply4D(point, result);
			return;
		}
	}

	{
		const RelativeRotationPose3D *realPose = dynamic_cast<const RelativeRotationPose3D*>(pose);
		if(realPose)
		{
			realPose->apply4D(point, result);
			return;
		}
	}

	{
		const RelativePose3D *realPose = dynamic_cast<const RelativePose3D*>(pose);
		if(realPose)
		{
			realPose->apply4D(point, result);
			return;
		}
	}

	assert(false);
}

template<class T>
void Pose3D::ApplyInv3D(const Pose3D *pose, const T * const point, T*result)
{
	{
		const FullPose3D *realPose = dynamic_cast<const FullPose3D*>(pose);
		if(realPose)
		{
			realPose->applyInv3D(point, result);
			return;
		}
	}

	{
		const RelativeRotationPose3D *realPose = dynamic_cast<const RelativeRotationPose3D*>(pose);
		if(realPose)
		{
			realPose->applyInv3D(point, result);
			return;
		}
	}

	{
		const RelativePose3D *realPose = dynamic_cast<const RelativePose3D*>(pose);
		if(realPose)
		{
			realPose->applyInv3D(point, result);
			return;
		}
	}

	assert(false);
}

template<class T>
void Pose3D::ApplyInv4D(const Pose3D *pose, const T * const point, T*result)
{
	{
		const FullPose3D *realPose = dynamic_cast<const FullPose3D*>(pose);
		if(realPose)
		{
			realPose->applyInv4D(point, result);
			return;
		}
	}

	{
		const RelativeRotationPose3D *realPose = dynamic_cast<const RelativeRotationPose3D*>(pose);
		if(realPose)
		{
			realPose->applyInv4D(point, result);
			return;
		}
	}

	{
		const RelativePose3D *realPose = dynamic_cast<const RelativePose3D*>(pose);
		if(realPose)
		{
			realPose->applyInv4D(point, result);
			return;
		}
	}

	assert(false);
}

/////////////////////////////////////////////////////////////////////
//FullPose3D
template<class T>
void FullPose3D::Apply3D(const T * const poseR, const T * const poseT, const T * const point, T*result)
{
	auto Rmat = CeresUtils::FixedRowMajorAdapter3x3(poseR);
	auto pointmat = CeresUtils::FixedRowMajorAdapter3x1(point);
	auto resmat = CeresUtils::FixedRowMajorAdapter3x1(result);

	CeresUtils::matrixMatrix(Rmat, pointmat, resmat);
	result[0] += poseT[0];
	result[1] += poseT[1];
	result[2] += poseT[2];
}

template<class T>
void FullPose3D::apply3D(const T * const point, T*result) const
{
	CeresUtils::matrixPoint(mRotation, point, result);
	result[0] += T(mTranslation[0]);
	result[1] += T(mTranslation[1]);
	result[2] += T(mTranslation[2]);
}

template<class T>
void FullPose3D::apply4D(const T * const point, T*result) const
{
	CeresUtils::matrixPoint(mRotation, point, result);
	result[0] += point[3]*T(mTranslation[0]);
	result[1] += point[3]*T(mTranslation[1]);
	result[2] += point[3]*T(mTranslation[2]);
}

template<class T>
void FullPose3D::applyInv3D(const T * const point, T*result) const
{
	T xt[3];
	xt[0] = point[0] - T(mTranslation[0]);
	xt[1] = point[1] - T(mTranslation[1]);
	xt[2] = point[2] - T(mTranslation[2]);
	CeresUtils::matrixTranspPoint(mRotation, xt, result);
}

template<class T>
void FullPose3D::applyInv4D(const T * const point, T*result) const
{
	T xt[3];
	xt[0] = point[0] - point[3]*T(mTranslation[0]);
	xt[1] = point[1] - point[3]*T(mTranslation[1]);
	xt[2] = point[2] - point[3]*T(mTranslation[2]);
	CeresUtils::matrixTranspPoint(mRotation, xt, result);
}

template<class T>
void FullPose3D::MakeRelativePose(const T * const refR, const T * const refT, const T * const imgR, const T * const imgT, T *relR, T *relT)
{
	auto refRMat = CeresUtils::FixedRowMajorAdapter3x3(refR);
	auto imgRMat = CeresUtils::FixedRowMajorAdapter3x3(imgR);
	auto relRMat = CeresUtils::FixedRowMajorAdapter3x3(relR);

	auto refTVec = CeresUtils::FixedRowMajorAdapter3x1(refT);
	//auto imgTVec = CeresUtils::FixedRowMajorAdapter3x1(imgT);
	//auto relTVec = CeresUtils::FixedRowMajorAdapter3x1(relT);

	//Relative rotation
	//relR = imgR * refR.t();
	CeresUtils::matrixMatrixTransp(imgRMat, refRMat, relRMat);

	//Relative translation
	//relT = imgT - relR*refT;
	T temp[3];
	auto tempVec = CeresUtils::FixedRowMajorAdapter3x1(temp);
	CeresUtils::matrixMatrix(relRMat, refTVec, tempVec);
	relT[0] = imgT[0] - temp[0];
	relT[1] = imgT[1] - temp[1];
	relT[2] = imgT[2] - temp[2];
}

/////////////////////////////////////////////////////////////////////
//RelativeRotationPose3D
template<class T>
void RelativeRotationPose3D::apply3D(const T * const point, T*result) const
{
	T pb[3];
	Pose3D::Apply3D(mBasePose, point, pb);

	CeresUtils::matrixPoint(mRelativeRotation, point, result);
}

template<class T>
void RelativeRotationPose3D::apply4D(const T * const point, T*result) const
{
	T pb[3];
	Pose3D::Apply4D(mBasePose, point, pb);

	CeresUtils::matrixPoint(mRelativeRotation, point, result);
}

/////////////////////////////////////////////////////////////////////
//RelativePose3D
template<class T>
void RelativePose3D::apply3D(const T * const point, T*result) const
{
	T pb[3];
	Pose3D::Apply3D(mBasePose, point, pb);
	mRelativePose.apply3D(pb,result);
}

//Apply to ceres variables (point is in homogeneous 4D)
template<class T>
void RelativePose3D::apply4D(const T * const point, T*result) const
{
	T pb[4];
	Pose3D::Apply4D(mBasePose, point, pb);
	pb[3] = point[3];
	mRelativePose.apply4D(pb,result);
}

}

#endif //POSE3DCERES_H_
