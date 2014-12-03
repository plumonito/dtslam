/*
 * ReprojectionError3DImpl.hpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef REPROJECTIONERROR3DIMPL_HPP_
#define REPROJECTIONERROR3DIMPL_HPP_

namespace dtslam
{

template<class T>
void ReprojectionError3D::pointResiduals(const T &u, const T &v, const int i, T *residuals) const
{
	const cv::Point2f &p = mImgPoints[i];

	residuals[0] = (u-T(p.x))/T(mScale);
	residuals[1] = (v-T(p.y))/T(mScale);
}

template<class T>
void ReprojectionError3D::computeAllResidualsXc(const T * const xc, T *allResiduals) const
{
	if(CeresUtils::ToDouble(xc[2]) <= 0)
	{
		//Negative point
		for(int i=0; i<2*mImagePointCount; ++i)
			allResiduals[i] = T(1e3);
	}
	else
	{
		//Point in front of camera, proceed
		T u,v;
		mCamera->projectFromWorld(xc[0],xc[1],xc[2],u,v);

		//Calculate residuals for all points
		for(int i=0; i<mImagePointCount; ++i)
		{
			pointResiduals(u,v,i,allResiduals+2*i);
		}
	}
}

template<class T>
void ReprojectionError3D::computeAllResidualsRparams(const T * const rparams, const T * const t, const T * const x, T *allResiduals) const
{
	T xr[3];
	ceres::AngleAxisRotatePoint(rparams, x, xr);

	T xc[3];
	xc[0] = xr[0]+t[0];
	xc[1] = xr[1]+t[1];
	xc[2] = xr[2]+t[2];

	computeAllResidualsXc(xc,allResiduals);
}

template<class T>
void ReprojectionError3D::computeAllResidualsRmat(const T * const R, const T * const t, const T * const x, T *allResiduals) const
{
	auto xwmat = CeresUtils::FixedRowMajorAdapter3x1(x);

	auto Rmat = CeresUtils::FixedRowMajorAdapter3x3(R);
	T xr[3];
	auto xrmat = CeresUtils::FixedRowMajorAdapter3x1(xr);
	CeresUtils::matrixMatrix(Rmat,xwmat,xrmat);

	T xc[3];
	xc[0] = xr[0]+t[0];
	xc[1] = xr[1]+t[1];
	xc[2] = xr[2]+t[2];

	computeAllResidualsXc(xc,allResiduals);
}

template<class T>
void ReprojectionError3D::residualsToErrors(const std::vector<T> &allResiduals, const float errorThreshold, MatchReprojectionErrors &errors) const
{
	assert(allResiduals.size() == 2*mImagePointCount);

	float errorThresholdSq = errorThreshold*errorThreshold;

	//Init errors
	errors.isInlier=false;
	errors.bestReprojectionErrorSq = std::numeric_limits<float>::infinity();
	errors.reprojectionErrorsSq.resize(mImagePointCount);
	errors.isImagePointInlier.resize(mImagePointCount);
	for(int i=0; i!=mImagePointCount; ++i)
	{
		float r1 = (float)allResiduals[2 * i];
		float r2 = (float)allResiduals[2 * i + 1];

		auto &errorSq = errors.reprojectionErrorsSq[i];
		errorSq = r1*r1+r2*r2;

		//Min
		if(errorSq < errors.bestReprojectionErrorSq)
			errors.bestReprojectionErrorSq = errorSq;

		//Update errors
		if(errorSq < errorThresholdSq)
		{
			errors.isImagePointInlier[i] = true;
			errors.isInlier = true;
		}
		else
			errors.isImagePointInlier[i] = false;
	}
}


template<class T>
bool PoseReprojectionError3D::operator()(const T * const rparams, const T * const t, T *residuals) const
{
	T x[3] = {T(mFeaturePosition[0]),T(mFeaturePosition[1]),T(mFeaturePosition[2])};

	if(mImagePointCount==1)
	{
		computeAllResidualsRparams(rparams, t, x, residuals);
	}
	else
	{
		std::vector<T> allResiduals(2*mImagePointCount);
		computeAllResidualsRparams(rparams, t, x, allResiduals.data());

		int minIndex;
		CeresUtils::GetMinResiduals<2>(allResiduals.data(), mImagePointCount, residuals, minIndex);
	}
	return true;
}

template<class T>
bool BAReprojectionError3D::operator()(const T * const rparams, const T * const t, const T * const x, T *residuals) const
{
	if(mImagePointCount==1)
	{
		computeAllResidualsRparams(rparams, t, x, residuals);
	}
	else
	{
		std::vector<T> allResiduals(2*mImagePointCount);
		computeAllResidualsRparams(rparams, t, x, allResiduals.data());

		int minIndex;
		CeresUtils::GetMinResiduals<2>(allResiduals.data(), mImagePointCount, residuals, minIndex);
	}
	return true;
}

}



#endif /* REPROJECTIONERROR3DIMPL_HPP_ */
