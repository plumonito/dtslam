/*
 * EpipolarSegmentErrorImpl.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

namespace dtslam
{

/////////////////////////////////////////////////////////////////////////////////////////////////
//EpipolarSegmentError
/////////////////////////////////////////////////////////////////////////////////////////////////
	
template<class T>
void EpipolarSegmentError::computeAllResiduals(const T * const refR, const T * const refT, const T * const imgR, const T * const imgT, T *allResiduals) const
{
	//Find relative pose between frames
	T relR[9];
	T relT[3];
	FullPose3D::MakeRelativePose(refR, refT, imgR, imgT, relR, relT);

	//Get minimum depth img X
	T refMinDepthX[3] = {T(mRefMinDepthX.x), T(mRefMinDepthX.y), T(mRefMinDepthX.z)};
	T imgMinDepthX[3];
	FullPose3D::Apply3D(relR, relT, refMinDepthX, imgMinDepthX);

	//Normalize
	T norm = T(ceres::sqrt(CeresUtils::NormSqT<3>(imgMinDepthX)));
	T imgMinDepthXn[3];
	imgMinDepthXn[0] = imgMinDepthX[0]/norm;
	imgMinDepthXn[1] = imgMinDepthX[1]/norm;
	imgMinDepthXn[2] = imgMinDepthX[2]/norm;

	//Get infinite depth img X
	T refXn[3] = {T(mRefXn.x), T(mRefXn.y), T(mRefXn.z)};
	T imgInfiniteXn[3];
	CeresUtils::matrixMatrix(
			CeresUtils::FixedRowMajorAdapter3x3(relR),
			CeresUtils::FixedRowMajorAdapter3x1(refXn),
			CeresUtils::FixedRowMajorAdapter3x1(imgInfiniteXn));

	//Get epipolar line direction
	double lineDir[3];
	lineDir[0] = CeresUtils::ToDouble(imgInfiniteXn[0])-CeresUtils::ToDouble(imgMinDepthXn[0]);
	lineDir[1] = CeresUtils::ToDouble(imgInfiniteXn[1])-CeresUtils::ToDouble(imgMinDepthXn[1]);
	lineDir[2] = CeresUtils::ToDouble(imgInfiniteXn[2])-CeresUtils::ToDouble(imgMinDepthXn[2]);

	//Get error for each point
	for(int i=0; i!=mPointCount; ++i)
	{
		auto &imgXnFloat = mImgXns->at(i);
		T imgXn[3] = {T(imgXnFloat.x), T(imgXnFloat.y), T(imgXnFloat.z)};

		T distXn[3]; //This will contain the error in xn space

		//Calculate distance to infinite point
		distXn[0] = imgXn[0]-imgInfiniteXn[0];
		distXn[1] = imgXn[1]-imgInfiniteXn[1];
		distXn[2] = imgXn[2]-imgInfiniteXn[2];

		//Check if it is beyond infinite
		double signA = CeresUtils::ToDouble(distXn[0])*lineDir[0] + CeresUtils::ToDouble(distXn[1])*lineDir[1] + CeresUtils::ToDouble(distXn[2])*lineDir[2];
		if(signA < 0)
		{
			//Not beyond
			//Calculate distance to min depth point
			distXn[0] = imgXn[0]-imgMinDepthXn[0];
			distXn[1] = imgXn[1]-imgMinDepthXn[1];
			distXn[2] = imgXn[2]-imgMinDepthXn[2];

			//Now check if it is before min depth
			double signB = CeresUtils::ToDouble(distXn[0])*lineDir[0] + CeresUtils::ToDouble(distXn[1])*lineDir[1] + CeresUtils::ToDouble(distXn[2])*lineDir[2];
			if(signB > 0)
			{
				//Not beyond
				//Calculate distance to the epipolar plane
				T planeNormal[3];
				ceres::CrossProduct(imgMinDepthXn, imgInfiniteXn, planeNormal); //planeNormal is not normalized yet
				T normSq = CeresUtils::NormSqT<3>(planeNormal);
				T distanceToPlane = ceres::DotProduct(planeNormal, imgXn) / normSq; //The squared denominator normalizes the next equation
				distXn[0] = planeNormal[0]*distanceToPlane;
				distXn[1] = planeNormal[1]*distanceToPlane;
				distXn[2] = planeNormal[2]*distanceToPlane;
			}
		}

		//Now convert to pixel units
		auto &ujac = mUJac[i];
		auto &vjac = mVJac[i];

		allResiduals[2*i+0] = T(ujac[0])*distXn[0] + T(ujac[1])*distXn[1] + T(ujac[2])*distXn[2];
		allResiduals[2*i+1] = T(vjac[0])*distXn[0] + T(vjac[1])*distXn[1] + T(vjac[2])*distXn[2];
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////
//EpipolarSegmentErrorForPose
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void EpipolarSegmentErrorForPose::computeAllResiduals(const T * const imgRparams, const T * const imgT, T *allResiduals) const
{
	//Convert params to matrix
	T imgR[9];
	auto imgRMat = CeresUtils::FixedRowMajorAdapter3x3(imgR);
	ceres::AngleAxisToRotationMatrix(imgRparams, imgRMat);

	//Copy ref
	T refR[9];
	T refT[3];
	CeresUtils::copyMat(mRefR, CeresUtils::FixedRowMajorAdapter3x3(refR));
	CeresUtils::copyMat(mRefT, CeresUtils::FixedRowMajorAdapter3x1(refT));

	EpipolarSegmentError::computeAllResiduals(refR, refT, imgR, imgT, allResiduals);
}

inline void EpipolarSegmentErrorForPose::computeAllResiduals(const cv::Matx33f &imgR, const cv::Vec3f &imgT, std::vector<float> &allResiduals) const
{
	allResiduals.resize(kResidualsPerItem*mPointCount);
	EpipolarSegmentError::computeAllResiduals(mRefR.val, mRefT.val, imgR.val, imgT.val, allResiduals.data());
}

template<class T>
bool EpipolarSegmentErrorForPose::operator()(const T * const imgRparams, const T * const imgT, T * residuals) const
{
	//Residuals
	if(mPointCount==1)
	{
		computeAllResiduals(imgRparams, imgT, residuals);
	}
	else
	{
		std::vector<T> allResiduals(kResidualsPerItem*mPointCount);
		computeAllResiduals(imgRparams, imgT, allResiduals.data());

		int dummy;
		CeresUtils::GetMinResiduals<kResidualsPerItem>(allResiduals.data(), mPointCount, residuals, dummy);
	}

	return true;
}


/////////////////////////////////////////////////////////////////////////////////////////////////
//EpipolarSegmentErrorForBA
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
void EpipolarSegmentErrorForBA::computeAllResiduals(const T * const refRparams, const T * const refT, const T * const imgRparams, const T * const imgT, T *allResiduals) const
{
	//Convert params to matrix
	T imgR[9];
	auto imgRMat = CeresUtils::FixedRowMajorAdapter3x3(imgR);
	ceres::AngleAxisToRotationMatrix(imgRparams, imgRMat);

	T refR[9];
	auto refRMat = CeresUtils::FixedRowMajorAdapter3x3(refR);
	ceres::AngleAxisToRotationMatrix(refRparams, refRMat);

	EpipolarSegmentError::computeAllResiduals(refR, refT, imgR, imgT, allResiduals);
}

template<class T>
bool EpipolarSegmentErrorForBA::operator()(const T * const refRparams, const T * const refT, const T * const imgRparams, const T * const imgT, T * residuals) const
{
	//Residuals
	if (mPointCount == 1)
	{
		computeAllResiduals(refRparams, refT, imgRparams, imgT, residuals);
	}
	else
	{
		std::vector<T> allResiduals(kResidualsPerItem*mPointCount);
		computeAllResiduals(refRparams, refT, imgRparams, imgT, allResiduals.data());

		int dummy;
		CeresUtils::GetMinResiduals<kResidualsPerItem>(allResiduals.data(), mPointCount, residuals, dummy);
	}

	return true;
}

} /* namespace nvslam */
