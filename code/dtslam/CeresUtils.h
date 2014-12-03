/*
 * CeresUtils.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef CERESUTILS_H_
#define CERESUTILS_H_

#include <ceres/jet.h>
#include <ceres/rotation.h>
#include <opencv2/core.hpp>
#include "PoseEstimationCommon.h"

namespace dtslam
{

//Unlike ceres, this is always row-major
template<typename T, int TRows, int TCols>
struct FixedMatrixAdapter: public ceres::MatrixAdapter<T,TCols,1> {
  explicit FixedMatrixAdapter(T* pointer)
    : ceres::MatrixAdapter<T,TCols,1>(pointer)
  {}

  const int kRows = TRows;
  const int kCols = TCols;
};

class CeresUtils
{
public:
	template <typename T>
	static FixedMatrixAdapter<T, 3, 3> FixedRowMajorAdapter3x3(T* pointer) {
	  return FixedMatrixAdapter<T, 3, 3>(pointer);
	}
	template <typename T>
	static FixedMatrixAdapter<T, 3, 1> FixedRowMajorAdapter3x1(T* pointer) {
	  return FixedMatrixAdapter<T, 3, 1>(pointer);
	}
	template <typename T>
	static FixedMatrixAdapter<T, 1, 3> FixedRowMajorAdapter1x3(T* pointer) {
	  return FixedMatrixAdapter<T, 1, 3>(pointer);
	}

	template<typename T, int N>
	static double ToDouble(const ceres::Jet<T,N> &val) {return val.a;}

	static double ToDouble(const double &val) {return val;}

	static double ToDouble(const float &val) {return val;}

	template<int N, typename T>
	static double NormSq(const T * const vec)
	{
		double val = ToDouble(vec[0]);
		double res = val*val;
		for(int i=1; i<N; ++i)
		{
			double val = ToDouble(vec[i]);
			res += val*val;
		}
		return res;
	}

	template<int N, typename T>
	static T NormSqT(const T * const vec)
	{
		T res = vec[0]*vec[0];
		for(int i=1; i<N; ++i)
		{
			res += vec[i]*vec[i];
		}
		return res;
	}


	template<class T, int TRows, int TCols>
	static void copyMat(const cv::Matx<float, TRows, TCols> &src, const FixedMatrixAdapter<T,TRows,TCols> &dst)
	{
		for(int j=0; j<TRows; ++j)
			for(int i=0; i<TCols; ++i)
			{
				dst(j,i) = T(src(j,i));
			}
	}

	template<class T, int TRows, int TCols>
	static void copyMat(const FixedMatrixAdapter<T,TRows,TCols> &src, cv::Matx<float, TRows, TCols> &dst)
	{
		for(int j=0; j<TRows; ++j)
			for(int i=0; i<TCols; ++i)
			{
				dst(j,i) = (float)ToDouble(src(j,i));
			}
	}


	template<class T, class MT, int Mm, int Mn>
	static void matrixPoint(const cv::Matx<MT,Mm,Mn> &mat, const T * const p, T * res)
	{
		for(int j=0; j<Mm; ++j)
		{
			res[j] = T(mat(j,0))*p[0];
			for(int i=1; i<Mn; ++i)
				res[j] += T(mat(j,i))*p[i];
		}
	}

	template<class T, class MT, int Mm, int Mn>
	static void matrixTranspPoint(const cv::Matx<MT,Mm,Mn> &mat, const T * const p, T * res)
	{
		for(int i=0; i<Mn; ++i)
		{
			res[i] = T(mat(0,i))*p[0];
			for(int j=1; j<Mm; ++j)
				res[i] += T(mat(j,i))*p[i];
		}
	}

	template<class MTa, class MTb, class MT, int TRows, int TInner, int TCols>
	static void matrixMatrix(const FixedMatrixAdapter<MTa,TRows,TInner> &matA, const FixedMatrixAdapter<MTb,TInner,TCols> &matB, const FixedMatrixAdapter<MT,TRows,TCols> &matRes)
	{
		for(int j=0; j<TRows; ++j)
			for(int i=0; i<TCols; ++i)
			{
				matRes(j,i) = matA(j,0)*matB(0,i);
				for(int k=1; k<TInner; ++k)
					matRes(j,i) += matA(j,k)*matB(k,i);
			}
	}

	template<class MTa, class MTb, class MT, int TRows, int TInner, int TCols>
	static void matrixMatrixTransp(const FixedMatrixAdapter<MTa,TRows,TInner> &matA, const FixedMatrixAdapter<MTb,TCols,TInner> &matB, FixedMatrixAdapter<MT,TRows,TCols> &matRes)
	{
		for(int j=0; j<TRows; ++j)
			for(int i=0; i<TCols; ++i)
			{
				matRes(j,i) = matA(j,0)*matB(i,0);
				for(int k=1; k<TInner; ++k)
					matRes(j,i) += matA(j,k)*matB(i,k);
			}
	}

	template<int ResidualsPerItem, class T>
	static void GetMinResiduals(const T * const allResiduals, int itemCount, T *residuals, int &minIndex)
	{
		double minSq = CeresUtils::NormSq<ResidualsPerItem>(allResiduals);
		minIndex = 0;
		for(int k=0; k<ResidualsPerItem; ++k)
			residuals[k] = allResiduals[k];

		auto itemPtr = allResiduals+ResidualsPerItem;
		for(int i=1; i!=itemCount; ++i)
		{
			const double sq = CeresUtils::NormSq<ResidualsPerItem>(itemPtr);
			if(sq<minSq)
			{
				minSq = sq;
				minIndex = i;
				for(int k=0; k<ResidualsPerItem; ++k)
					residuals[k] = itemPtr[k];
			}
			itemPtr += ResidualsPerItem;
		}
	}

	template<int ResidualsPerItem, class T>
	static void ResidualsToErrors(int itemCount, const std::vector<T> &allResiduals, const float errorThresholdSq, MatchReprojectionErrors &errors)
	{
		assert(allResiduals.size() == ResidualsPerItem*itemCount);

		//Init errors
		errors.isInlier=false;
		errors.bestReprojectionErrorSq = std::numeric_limits<float>::infinity();
		errors.reprojectionErrorsSq.resize(itemCount);
		errors.isImagePointInlier.resize(itemCount);
		for(int i=0; i!=itemCount; ++i)
		{
			auto &errorSq = errors.reprojectionErrorsSq[i];
			float r0 = (float)allResiduals[2 * i];
			errorSq = r0*r0;
			for(int j=1; j<ResidualsPerItem; ++j)
			{
				float rj = (float)allResiduals[2 * i + j];
				errorSq += rj*rj;
			}

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

};

}

#endif /* CERESUTILS_H_ */
