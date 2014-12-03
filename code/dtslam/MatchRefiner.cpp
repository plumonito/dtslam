/*
 * MatchRefiner.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "MatchRefiner.h"
#include <opencv2/imgproc.hpp>
#include <Eigen/Eigen>
#include "PatchWarper.h"
#include "Profiler.h"

namespace dtslam
{

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This version uses manual loops. It doesn't seem optimal but it's the fastest so far.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
MatchRefiner::MatchRefiner()
{
	mRefJacobian.create(kPatchSize, kPatchSize);
}

void MatchRefiner::setRefPatch(const cv::Mat1b *refPatch)
{
	assert(refPatch->isContinuous());
	assert(refPatch->step[0] == kPatchSize);

	//Copy
	mRefPatch = refPatch;

	//Build jacobian and hessian
	cv::Vec6i hessianTriangle(0, 0, 0, 0, 0, 0);
	const int rowStride = kPatchSize;

	const unsigned char *patchPtr = (*mRefPatch)[0] + rowStride + 1;
	cv::Vec2i *jacPtr = mRefJacobian[0] + rowStride + 1;

	for (int j = 1; j < kPatchSize - 1; j++)
	{
		for (int i = 1; i < kPatchSize - 1; i++)
		{
			//Jacobian
			cv::Vec2i &jac = *jacPtr;
			jac[0] = ((int)patchPtr[1] - (int)patchPtr[-1]) / 2; //dx
			jac[1] = ((int)patchPtr[rowStride] - (int)patchPtr[-rowStride]) / 2; //dy

			//Lower left triangle of Hessian (=J'*J)
			//Note that jac[2]==1
			hessianTriangle[0] += jac[0] * jac[0];
			hessianTriangle[1] += jac[0] * jac[1];
			hessianTriangle[2] += jac[1] * jac[1];
			hessianTriangle[3] += jac[0];
			hessianTriangle[4] += jac[1];
			hessianTriangle[5] += 1;

			patchPtr++;
			jacPtr++;
		}
		patchPtr += 2;
		jacPtr += 2;
	}

	//Build hessian
	cv::Matx33f hessian;
	for (int v = 0, j = 0; j < 3; j++)
	{
		for (int i = 0; i <= j; i++)
		{
			hessian(j, i) = hessian(i, j) = (float)hessianTriangle[v++];
		}
	}
	mRefHessianInv = hessian.inv(cv::DECOMP_CHOLESKY);

	//Calculate variance of patch
	//    const uchar *refPtr = (*mRefPatch)[0];
	//    uint32_t refSumSq=0;
	//    uint32_t refSum=0;
	//
	//    for (int j = 0; j < kPatchSize; j++)
	//    {
	//        for (int i = 0; i < kPatchSize; i++)
	//        {
	//            refSumSq += refPtr[0]*refPtr[0];
	//            refSum += refPtr[0];
	//
	//            //Advance col
	//            refPtr++;
	//        }
	//        //Advance row
	//    }
	//    uint32_t refSqSum = refSum*refSum;
	//	mRefVar = (refSumSq >> (kPatchSizeBits*2)) - (refSqSum >> (kPatchSizeBits*4));
}

void MatchRefiner::setImg(const cv::Mat1b *img)
{
	mImg = img;
}

bool MatchRefiner::refine()
{
	//const float kConvergenceLimit = 0.03f;
	const float kConvergenceLimit = 0.05f;

	//Prepare patch matrix
	mImgPatch.release();
	mImgPatch.create(kPatchSize, kPatchSize);
	assert(mImgPatch.isContinuous());


	//Initial transform
	mMeanDiff = 0;

	//Iterate
	for (mIterationCount = 0; mIterationCount < kMaxIterations; ++mIterationCount)
	{
		//Refine
		float updateSq = refineIteration();

		//Check bounds
		if (mOrigin.x < 0 || mOrigin.y < 0 || mOrigin.x >= mImg->cols - 1 - kPatchSize || mOrigin.y >= mImg->rows - 1 - kPatchSize)
		{
			mOrigin.x = 0;
			mOrigin.y = 0;
			return false;
		}

		//Check convergence
		if (updateSq < kConvergenceLimit)
			return true;
	}

	return true;
}

float MatchRefiner::refineIteration()
{
	if (!buildImgPatch())
		return 0;

	//bool log=false;
	//if(log)
	//{
	//	DTSLAM_LOG << "Iteration=" << mIterationCount << "\n";
	//	DTSLAM_LOG << "img=" << mImgPatch << "\n";
	//}

	//Loop over pixels
	const size_t rowStride = kPatchSize;

	const uchar *refPtr = (*mRefPatch)[0] + rowStride + 1;
	const uchar *imgPtr = mImgPatch[0] + rowStride + 1;
	const cv::Vec2i *refJacPtr = mRefJacobian[0] + rowStride + 1;

	cv::Vec3i accum(0, 0, 0); // sum(J(p)*diff)

	for (int j = 1; j < kPatchSize - 1; j++)
	{
		for (int i = 1; i < kPatchSize - 1; i++)
		{
			const cv::Vec2i &jac = *refJacPtr;

			int imgValue = *imgPtr;
			int refValue = *refPtr;
			int diff = imgValue - refValue + mMeanDiff;

			accum[0] += diff * jac[0];
			accum[1] += diff * jac[1];
			accum[2] += diff;

			//Advance col
			refJacPtr++;
			refPtr++;
			imgPtr++;
		}

		//Advance row
		refJacPtr += 2;
		refPtr += 2;
		imgPtr += 2;
	}

	//Calculate update
	const cv::Vec3f update = mRefHessianInv * (cv::Vec3f)accum;
	mOrigin.x -= update[0];
	mOrigin.y -= update[1];
	mMeanDiff -= (int)update[2];

	//float updateSquared = cvutils::PointDistSq(lastOrigin, mOrigin);
	float updateSquared = abs(update[0]) + abs(update[1]);
	return updateSquared;
}

bool MatchRefiner::buildImgPatch()
{
	cv::Point2i iorigin = cv::Point2i((int)mOrigin.x, (int)mOrigin.y);
	if (iorigin.x < 0 || iorigin.y < 0 || iorigin.x >= mImg->cols - 1 - kPatchSize || iorigin.y >= mImg->rows - 1 - kPatchSize)
		return false;

	const float dx = mOrigin.x - iorigin.x;
	const float dy = mOrigin.y - iorigin.y;
	const float mixTL = (1.0f - dx) * (1.0f - dy);
	const float mixTR = (dx)* (1.0f - dy);
	const float mixBL = (1.0f - dx) * (dy);
	const float mixBR = (dx)* (dy);

	const size_t imgRowStride = mImg->step[0];

	uchar *patchPtr = mImgPatch[0];

	//Make patch
	for (int j = 0; j < kPatchSize; j++)
	{
		const uchar *imgPtr = (*mImg)[iorigin.y + j] + (int)iorigin.x;
		for (int i = 0; i < kPatchSize; i++)
		{
			float value = mixTL * imgPtr[0] + mixTR * imgPtr[1] + mixBL * imgPtr[imgRowStride]
				+ mixBR * imgPtr[imgRowStride + 1];
			*patchPtr = (uchar)value;

			//Advance col
			patchPtr++;
			imgPtr++;
		}
	}
	return true;
}

int MatchRefiner::getScore()
{
	//Check image bounds
	const cv::Point2f maxOrigin((float)mImg->cols - kPatchSize - 1, (float)mImg->rows - kPatchSize - 1);
	if (mOrigin.x < 0 || mOrigin.x >= maxOrigin.x
		|| mOrigin.y < 0 || mOrigin.y >= maxOrigin.y)
		return std::numeric_limits<int>::max();

	if (mImgPatch.empty())
	{
		mImgPatch.create(kPatchSize, kPatchSize);
		buildImgPatch();
	}

	unsigned int score = 0;

	//    //Debug: This calculates the mean of each patch
	//    float meanImg=0;
	//    float meanRef=0;
	//    {
	//        const uchar *refPtr = (*mRefPatch)[0];
	//        const uchar *imgPtr = mImgPatch[0];
	//
	//        for (int j = 0; j < kPatchSize; j++)
	//        {
	//            for (int i = 0; i < kPatchSize; i++)
	//            {
	//                meanImg += (*imgPtr);
	//                meanRef += (*refPtr);
	//
	//                //Advance col
	//                refPtr++;
	//                imgPtr++;
	//            }
	//
	//            //Advance row
	//        }
	//        meanImg /= kPatchSize*kPatchSize;
	//        meanRef /= kPatchSize*kPatchSize;
	//    }

	const uchar *refPtr = (*mRefPatch)[0];
	const uchar *imgPtr = mImgPatch[0];
	int iMeanDiff = mMeanDiff;

	for (int j = 0; j < kPatchSize; j++)
	{
		for (int i = 0; i < kPatchSize; i++)
		{
			int imgValue = *imgPtr;
			int refValue = *refPtr;
			int diff = imgValue - refValue + iMeanDiff;

			//Debug: this is the real measure, using mMeanDiff is a shortcut but accurate enough
			//float diff = (*imgPtr) - meanImg - (*refPtr) + meanRef;

			score += diff*diff;

			//Advance col
			refPtr++;
			imgPtr++;
		}

		//Advance row
	}

	//score >>= 2*kPatchSizeBits;

	//    ZssdComparer c;
	//    c.setRefPatch(*mRefPatch);
	//    auto res = c.compare(mImgPatch);
	//    DTSLAM_LOG << "Score according to zssdcompare=" << res << "\n";

	//score = (score<<6) / mRefVar;
	//DTSLAM_LOG << "Score according to refiner=" << scorei << "\n";

	//DTSLAM_LOG << "ref=" << *mRefPatch << "\n";
	//DTSLAM_LOG << "img=" << mImgPatch << "\n";
	return score;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This version uses Eigen. I was hoping it'd be faster than manual loops but it turned out to be slower
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//MatchRefiner::MatchRefiner()
//{
//}
//
//void MatchRefiner::setRefPatch(const cv::Mat1b *refPatch)
//{
//	assert(refPatch->isContinuous());
//	assert(refPatch->step[0] == kPatchSize);
//
//	//Copy
//	Eigen::Map<Eigen::Array<unsigned char, kPatchSize, kPatchSize, Eigen::RowMajor>> refPatchEigen(refPatch->data);
//	mRefPatch = refPatchEigen.cast<short>();
//
//	//Jacobian
//	mRefJacobianX.setZero();
//	mRefJacobianY.setZero();
//	mRefJacobianX.block<kPatchSize - 2, kPatchSize - 2>(1, 1) = (mRefPatch.block<kPatchSize - 2, kPatchSize - 2>(1, 2) - mRefPatch.block<kPatchSize - 2, kPatchSize - 2>(1, 0)).cast<int>() / 2;
//	mRefJacobianY.block<kPatchSize - 2, kPatchSize - 2>(1, 1) = (mRefPatch.block<kPatchSize - 2, kPatchSize - 2>(2, 1) - mRefPatch.block<kPatchSize - 2, kPatchSize - 2>(0, 1)).cast<int>() / 2;
//
//	//Hessian
//	cv::Vec6i hessianTriangle(0, 0, 0, 0, 0, 0);
//	hessianTriangle[0] = (mRefJacobianX * mRefJacobianX).sum(); //sum(dx*dx)
//	hessianTriangle[1] = (mRefJacobianX * mRefJacobianY).sum(); //sum(dx*dy)
//	hessianTriangle[2] = (mRefJacobianY * mRefJacobianY).sum(); //sum(dy*dy)
//	hessianTriangle[3] = mRefJacobianX.sum(); //sum(dx)
//	hessianTriangle[4] = mRefJacobianY.sum(); //sum(dy)
//	hessianTriangle[5] = (kPatchSize - 2)*(kPatchSize - 2);
//
//    //Build hessian
//	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> hessian;
//	for (int v = 0, j = 0; j < 3; j++)
//    {
//        for(int i = 0; i <= j; i++)
//        {
//            hessian(j, i) = hessian(i, j) = (float)hessianTriangle[v++];
//        }
//    }
//
//	mRefHessianInv = hessian.inverse();
//}
//
//void MatchRefiner::setImg(const cv::Mat1b *img)
//{
//	mImg = img;
//}
//
//bool MatchRefiner::refine()
//{
//	//const float kConvergenceLimit = 0.03f;
//	const float kConvergenceLimit = 0.05f;
//
//    //Prepare patch matrix
//	mValidImgPatch = false;
//
//    //Initial transform
//    mMeanDiff = 0;
//
//    //Iterate
//    for(mIterationCount = 0; mIterationCount < kMaxIterations; ++mIterationCount)
//    {
//    	//Refine
//    	float updateSq = refineIteration();
//
//		//Check bounds
//		if (mOrigin.x < 0 || mOrigin.y < 0 || mOrigin.x >= mImg->cols - 1 - kPatchSize || mOrigin.y >= mImg->rows - 1 - kPatchSize)
//		{
//			mOrigin.x = 0;
//			mOrigin.y = 0;
//			return false;
//		}
//
//    	//Check convergence
//    	if(updateSq < kConvergenceLimit)
//    		return true;
//    }
//
//    return true;
//}
//
//float MatchRefiner::refineIteration()
//{
//    if(!buildImgPatch())
//    	return 0;
//
//	Eigen::Array<int, kPatchSize, kPatchSize, Eigen::RowMajor> diff;
//	diff = (mImgPatch - mRefPatch + mMeanDiff).cast<int>();
//
//	Eigen::Vector3i accum;
//	accum[0] = (diff * mRefJacobianX).sum(); //sum(diff*dx)
//	accum[1] = (diff * mRefJacobianY).sum(); //sum(diff*dy)
//	accum[2] = diff.sum(); //sum(diff)
//
//    //Calculate update
//	Eigen::Vector3f update = mRefHessianInv * accum.cast<float>();
//    mOrigin.x -= update[0];
//    mOrigin.y -= update[1];
//    mMeanDiff -= (int)update[2];
//
//    //float updateSquared = cvutils::PointDistSq(lastOrigin, mOrigin);
//	float updateSquared = abs(update[0])+abs(update[1]);
//    return updateSquared;
//}
//
//bool MatchRefiner::buildImgPatch()
//{
//    cv::Point2i iorigin = cv::Point2i((int)mOrigin.x, (int)mOrigin.y);
//	if (iorigin.x < 0 || iorigin.y < 0 || iorigin.x >= mImg->cols - 1 - kPatchSize || iorigin.y >= mImg->rows - 1 - kPatchSize)
//	{
//		mValidImgPatch = false;
//		return false;
//	}
//	
//	Eigen::Map<Eigen::Array<uchar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> imgMap(mImg->data, mImg->rows, mImg->cols);
//
//    const float dx = mOrigin.x - iorigin.x;
//    const float dy = mOrigin.y - iorigin.y;
//
//	auto blockLeft = imgMap.block<kPatchSize+1, kPatchSize>(iorigin.y, iorigin.x);
//	auto blockRight = imgMap.block<kPatchSize+1, kPatchSize>(iorigin.y, iorigin.x + 1);
//
//	Eigen::Array<uchar, kPatchSize+1, kPatchSize, Eigen::RowMajor> pass1;
//	pass1 = (blockLeft.cast<float>()*(1 - dx) + blockRight.cast<float>()*dx).cast<uchar>();
//	
//	auto blockUp = pass1.block<kPatchSize, kPatchSize>(0,0);
//	auto blockDown = pass1.block<kPatchSize, kPatchSize>(1,0);
//	mImgPatch = (blockUp.cast<float>()*(1 - dy) + blockDown.cast<float>()*dy).cast<short>();
//
//	mValidImgPatch = true;
//    return true;
//}
//
//int MatchRefiner::getScore()
//{
//    //Check image bounds
//	const cv::Point2f maxOrigin((float)mImg->cols - kPatchSize - 1, (float)mImg->rows - kPatchSize - 1);
//	if(mOrigin.x < 0 || mOrigin.x >= maxOrigin.x
//		|| mOrigin.y < 0 || mOrigin.y >= maxOrigin.y)
//		return std::numeric_limits<int>::max();
//
//	if(!mValidImgPatch)
//	{
//		buildImgPatch();
//	}
//
//	Eigen::Array<int, kPatchSize, kPatchSize, Eigen::RowMajor> diff;
//	diff = (mImgPatch - mRefPatch + mMeanDiff).cast<int>();
//	unsigned int score = (diff*diff).sum(); //sum(diff*diff)
//
//	return score;
//}

} /* namespace dtslam */
