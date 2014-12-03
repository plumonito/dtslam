/*
 * MatchRefiner.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef MATCHREFINER_H_
#define MATCHREFINER_H_

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include "PatchWarper.h"

namespace dtslam
{

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This version uses manual loops. It doesn't seem optimal but it's the fastest so far.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MatchRefiner
{
public:
	MatchRefiner();

	void setRefPatch(const cv::Mat1b *refPatch);
	void setImg(const cv::Mat1b *img);

	cv::Point2f getCenter() const {return cv::Point2f(mOrigin.x+kPatchCenterOffset, mOrigin.y+kPatchCenterOffset);}
	void setCenter(const cv::Point2f &center) {mOrigin = cv::Point2f(center.x-kPatchCenterOffset, center.y-kPatchCenterOffset);}

	const cv::Mat1b &getImgPatch() const {return mImgPatch;}

	bool refine();

	const cv::Mat1b &getImgPatch()
	{
		if(mImgPatch.empty())
		{
		    mImgPatch.create(kPatchSize, kPatchSize);
			buildImgPatch();
		}
		return mImgPatch;
	}
	int getScore();
	int getIterationCount() const {return mIterationCount;}

protected:
	static const int kPatchSizeBits=PatchWarper::kPatchSizeBits;
	static const int kPatchSize=PatchWarper::kPatchSize;
	static const int kPatchCenterOffset=3;

	static const int kMaxIterations=5;

	int mIterationCount;

	const cv::Mat1b *mRefPatch;
	cv::Mat2i mRefJacobian;
	cv::Matx33f mRefHessianInv;
	//unsigned int mRefVar;

	const cv::Mat1b *mImg;
	cv::Mat1b mImgPatch;

	int mMeanDiff;
	cv::Point2f mOrigin;

	float refineIteration();
	bool buildImgPatch();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This version uses Eigen. I was hoping it'd be faster than manual loops but it turned out to be slower
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//class MatchRefiner
//{
//public:
//	MatchRefiner();
//
//	void setRefPatch(const cv::Mat1b *refPatch);
//	void setImg(const cv::Mat1b *img);
//
//	cv::Point2f getCenter() const {return cv::Point2f(mOrigin.x+kPatchCenterOffset, mOrigin.y+kPatchCenterOffset);}
//	void setCenter(const cv::Point2f &center) {mOrigin = cv::Point2f(center.x-kPatchCenterOffset, center.y-kPatchCenterOffset);}
//
//	cv::Mat1s getImgPatch() const { return cv::Mat1s(kPatchSize, kPatchSize, const_cast<short*>(mImgPatch.data()), mImgPatch.rowStride()*sizeof(short)); }
//
//	bool refine();
//
//	cv::Mat1s getImgPatch()
//	{
//		if (!mValidImgPatch)
//		{
//			buildImgPatch();
//		}
//		return cv::Mat1s(kPatchSize, kPatchSize, mImgPatch.data(), mImgPatch.rowStride()*sizeof(short));
//	}
//	int getScore();
//	int getIterationCount() const {return mIterationCount;}
//
//protected:
//	static const int kPatchSizeBits=PatchWarper::kPatchSizeBits;
//	static const int kPatchSize=PatchWarper::kPatchSize;
//	static const int kPatchCenterOffset=3;
//
//	static const int kMaxIterations=5;
//
//	int mIterationCount;
//
//	Eigen::Array<short, kPatchSize, kPatchSize, Eigen::RowMajor> mRefPatch;
//	Eigen::Array<int, kPatchSize, kPatchSize, Eigen::RowMajor> mRefJacobianX;
//	Eigen::Array<int, kPatchSize, kPatchSize, Eigen::RowMajor> mRefJacobianY;
//	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> mRefHessianInv;
//	//unsigned int mRefVar;
//
//	const cv::Mat1b *mImg;
//	bool mValidImgPatch;
//	Eigen::Array<short, kPatchSize, kPatchSize, Eigen::RowMajor> mImgPatch;
//
//	int mMeanDiff;
//	cv::Point2f mOrigin;
//
//	float refineIteration();
//	bool buildImgPatch();
//};

} /* namespace dtslam */

#endif /* MATCHREFINER_H_ */
