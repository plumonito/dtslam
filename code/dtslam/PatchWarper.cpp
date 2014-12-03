/*
 * PatchWarper.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "PatchWarper.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "log.h"
#include "Profiler.h"
#include "cvutils.h"
#include "flags.h"


namespace dtslam {

///////////////////////////////////
// Implementation
PatchWarper::PatchWarper() :
	mMaxCornerDriftSq((float)(FLAGS_WarperMaxCornerDrift*FLAGS_WarperMaxCornerDrift))
{
}

cv::Mat PatchWarper::ExtractPatch(const cv::Mat &imageSrc, const cv::Point2f &centerSrc, const int octave, const int centerOffset, const int size)
{
	float scaleInv = 1.0f / (1<<octave);
	cv::Point2i tl = cv::Point2i((int)(scaleInv*centerSrc.x-centerOffset),(int)(scaleInv*centerSrc.y-centerOffset));

	//DTSLAM_LOG << "tl=" << tl << ",size=" << size << "\n";
	if(tl.x < 0 || tl.y < 0 || (tl.x+size)>=imageSrc.cols || (tl.y+size)>=imageSrc.rows)
		return cv::Mat(0,0,imageSrc.type());
	else
		return imageSrc(cv::Rect( tl, cv::Size(size, size)));
}

cv::Mat1b PatchWarper::ExtractPatch(const cv::Mat1b &imageSrc, const cv::Point2f &centerSrc, const cv::Point2f &centerDst, const int octave, const cv::Matx33f &homography, const int centerOffset, const int size)
{
	cv::Matx33f h2;
	float scale = (float)(1<<octave);
	float scaleInv = 1.0f/scale;
	cv::Matx33f hScale(scale,0,0, 0,scale,0, 0,0,1);
	//cv::Matx33f hScaleInv(scaleInv,0,0, 0,scaleInv,0, 0,0,1);
	//cv::Matx33f hOffset(1,0,-centerDst.x+centerOffset*scale, 0,1,-centerDst.y+centerOffset*scale, 0,0,1);
	cv::Matx33f hScaleInvAndOffset(scaleInv,0,-scaleInv*centerDst.x+centerOffset, 0,scaleInv,-scaleInv*centerDst.y+centerOffset, 0,0,1);

	h2 = hScaleInvAndOffset*homography*hScale;

	cv::Mat1b patchH(size,size);
	cv::warpPerspective(imageSrc,patchH,h2,patchH.size(),cv::INTER_NEAREST);

//	cv::Mat1b patch0 = ExtractPatch(imageSrc, centerSrc,octave,centerOffset,size);
//	cv::imwrite("patch0.png",patch0);
//	cv::imwrite("patchH.png",patchH);
	return patchH;
}

void PatchWarper::setSource(const cv::Mat1b *srcImage, const cv::Point2f &srcPosition)
{
	assert(srcImage!=NULL);

	if(mSourceImage == srcImage && mSourcePosition==srcPosition)
		return; //No change, ignore

	//Reset the warp
	mPatch = cv::Mat1b();

	//Update params
	mSourceCamera = NULL;
	mSourceImage = srcImage;
	mSourceScale = 1;
	mSourcePosition = srcPosition;
}
void PatchWarper::setSource(const CameraModel *srcCamera, const cv::Mat1b *srcImage, int srcOctave, const cv::Point2f &srcPosition, const cv::Point3f &srcXn)
{
	assert(srcCamera!=NULL);
	assert(srcImage!=NULL);

	if(mSourceImage != srcImage || mSourcePosition!=srcPosition)
	{
		//Image or center changed, reset patch
		mPatch = cv::Mat1b();
	}

	//Update params
	mSourceCamera = srcCamera;
	mSourceImage = srcImage;
	mSourceScale = 1<<srcOctave;
	mSourcePosition = srcPosition;
	mSourceXn = srcXn;
}

void PatchWarper::calculateWarp(const cv::Point2f &srcRightCenter, const cv::Point2f &srcCenterBottom)
{
	//Build affine
	const float factor = (float)(kPatchRightSize*mSourceScale);
	mWarpMatrix(0,0) = (srcRightCenter.x - mSourcePosition.x)/factor;
	mWarpMatrix(0,1) = (srcCenterBottom.x - mSourcePosition.x)/factor;
	mWarpMatrix(1,0) = (srcRightCenter.y - mSourcePosition.y)/factor;
	mWarpMatrix(1,1) = (srcCenterBottom.y - mSourcePosition.y)/factor;
	mWarpMatrix(0,2) = mSourcePosition.x/mSourceScale - mWarpMatrix(0,0)*kPatchCenterOffset - mWarpMatrix(0,1)*kPatchCenterOffset;
	mWarpMatrix(1,2) = mSourcePosition.y/mSourceScale - mWarpMatrix(1,0)*kPatchCenterOffset - mWarpMatrix(1,1)*kPatchCenterOffset;

	//This disables the warping
	//mWarpMatrix = cv::Matx23f::eye();
	//mWarpMatrix(0,2) = mSourcePosition.x/mSourceScale - kPatchCenterOffset;
	//mWarpMatrix(1,2) = mSourcePosition.y/mSourceScale - kPatchCenterOffset;

	//Warp corners to determine if we need to regenerate the map
	warpCorners(mWarpMatrix, mWarpedCorners);
}

void PatchWarper::calculateWarp(const cv::Matx33f &hDst2Src, const cv::Point2i &dstPosition)
{
	ProfileSection s("calculateWarp");

	float dstBottom = (float)(dstPosition.y + kPatchRightSize);
	float dstRight = (float)(dstPosition.x+kPatchRightSize);

	cv::Point2f srcCenterRight = cvutils::HomographyPoint(hDst2Src, cv::Point2f(dstRight, (float)dstPosition.y));
	cv::Point2f srcCenterBottom = cvutils::HomographyPoint(hDst2Src, cv::Point2f((float)dstPosition.x, dstBottom));

	calculateWarp(srcCenterRight, srcCenterBottom);
}

void PatchWarper::calculateWarp(const cv::Matx33f &dst2srcR, const CameraModel &dstCamera, const cv::Point2f &dstCenter, const cv::Point2f &dstRight, const cv::Point2f &dstBottom)
{
	assert(mSourceCamera);

	const cv::Point3f dstCenterXn = dstCamera.unprojectToWorld(dstCenter);
	const cv::Point3f worldCenterXn = dst2srcR*dstCenterXn;

	const cv::Point3f dstRightXn = dstCamera.unprojectToWorld(dstRight);
	const cv::Point3f worldRightXn = dst2srcR*dstRightXn;
	const cv::Point3f worldRightXnOffset = worldRightXn - worldCenterXn;
	const cv::Point3f srcRightXn = mSourceXn + worldRightXnOffset;
	const cv::Point2f srcRight = mSourceCamera->projectFromWorld(srcRightXn);

	const cv::Point3f dstBottomXn = dstCamera.unprojectToWorld(dstBottom);
	const cv::Point3f worldBottomXn = dst2srcR*dstBottomXn;
	const cv::Point3f worldBottomXnOffset = worldBottomXn - worldCenterXn;
	const cv::Point3f srcBottomXn = mSourceXn + worldBottomXnOffset;
	const cv::Point2f srcBottom = mSourceCamera->projectFromWorld(srcBottomXn);

	calculateWarp(srcRight, srcBottom);
}

void PatchWarper::calculateWarp(const cv::Matx33f &dst2srcR, const CameraModel &dstCamera, const cv::Point2f &dstCenter, const int dstScale)
{
	const cv::Point2f dstRight(dstCenter.x+dstScale*kPatchRightSize, dstCenter.y);
	const cv::Point2f dstBottom(dstCenter.x, dstCenter.y+dstScale*kPatchRightSize);

	calculateWarp(dst2srcR, dstCamera, dstCenter, dstRight, dstBottom);
}

void PatchWarper::calculateWarp(const cv::Matx33f &dst2srcR, const CameraModel &dstCamera, const cv::Matx23f &dst2cameraAffine, const cv::Point2f &dstCenter, const int dstScale)
{
	const cv::Point2f dstCenterS = cvutils::AffinePoint(dst2cameraAffine, dstCenter);

	const cv::Point2f dstRight(dstCenter.x+dstScale*kPatchRightSize, dstCenter.y);
	const cv::Point2f dstRightS = cvutils::AffinePoint(dst2cameraAffine, dstRight);

	const cv::Point2f dstBottom(dstCenter.x, dstCenter.y+dstScale*kPatchRightSize);
	const cv::Point2f dstBottomS = cvutils::AffinePoint(dst2cameraAffine, dstBottom);

	calculateWarp(dst2srcR, dstCamera, dstCenterS, dstRightS, dstBottomS);
}

void PatchWarper::calculateWarp3D(const Pose3D &srcPose, const cv::Vec3f &planeNormal, const cv::Vec3f &planePoint, const Pose3D &dstPose, const CameraModel &dstCamera, const cv::Matx23f &img2dstAffine, const cv::Point2f &imgCenter, const int imgScale)
{
	const cv::Point2f dstCenter = cvutils::AffinePoint(img2dstAffine, imgCenter);
	const cv::Point2f srcCenter = warpPoint3D(srcPose, planeNormal, planePoint, dstPose, dstCamera, dstCenter);

	const cv::Point2f imgRight(imgCenter.x + imgScale*kPatchRightSize, imgCenter.y);
	const cv::Point2f dstRight = cvutils::AffinePoint(img2dstAffine, imgRight);
	const cv::Point2f srcRight = warpPoint3D(srcPose, planeNormal, planePoint, dstPose, dstCamera, dstRight);
	const cv::Point2f srcRightDiff = srcRight - srcCenter;
	const cv::Point2f srcRightFixed = mSourcePosition + srcRightDiff;

	const cv::Point2f imgBottom(imgCenter.x, imgCenter.y + imgScale*kPatchRightSize);
	const cv::Point2f dstBottom = cvutils::AffinePoint(img2dstAffine, imgBottom);
	const cv::Point2f srcBottom = warpPoint3D(srcPose, planeNormal, planePoint, dstPose, dstCamera, dstBottom);
	const cv::Point2f srcBottomDiff = srcBottom - srcCenter;
	const cv::Point2f srcBottomFixed = mSourcePosition + srcBottomDiff;

	calculateWarp(srcRightFixed, srcBottomFixed);
}

cv::Point2f PatchWarper::warpPoint3D(const Pose3D &srcPose, const cv::Vec3f &planeNormal, const cv::Vec3f &planePoint, const Pose3D &dstPose, const CameraModel &dstCamera, const cv::Point2f &dstPoint)
{
	const cv::Point3f dstXn = dstCamera.unprojectToWorld(dstPoint);
	
	const cv::Point3f worldDirection = dstPose.getRotation().t() * dstXn;
	const cv::Point3f worldCenter = dstPose.getCenter();

	const cv::Point3f worldPoint = cvutils::linePlaneIntersection(worldCenter, worldDirection, planePoint, planeNormal);
	return mSourceCamera->projectFromWorld(srcPose.apply(worldPoint));
}

void PatchWarper::warpCorners(const cv::Matx23f &warp, cv::Point2f warpedCorners[4])
{
	//(0,0)
	warpedCorners[0].x = mWarpMatrix(0,2);
	warpedCorners[0].y = mWarpMatrix(1,2);
	//(0,kPatchSize)
	warpedCorners[1].x = mWarpMatrix(0,1)*kPatchSize + mWarpMatrix(0,2);
	warpedCorners[1].y = mWarpMatrix(1,1)*kPatchSize + mWarpMatrix(1,2);
	//(kPatchSize,0)
	warpedCorners[2].x = mWarpMatrix(0,0)*kPatchSize + mWarpMatrix(0,2);
	warpedCorners[2].y = mWarpMatrix(1,0)*kPatchSize + mWarpMatrix(1,2);
	//(kPatchSize,kPatchSize)
	warpedCorners[3].x = mWarpMatrix(0, 0)*kPatchSize + mWarpMatrix(0, 1)*kPatchSize + mWarpMatrix(0, 2);
	warpedCorners[3].y = mWarpMatrix(1, 0)*kPatchSize + mWarpMatrix(1, 1)*kPatchSize + mWarpMatrix(1, 2);
}

bool PatchWarper::patchNeedsUpdate()
{
	if(mPatch.empty())
		return true;

	for(int i=0; i<4; i++)
	{
		const float dx = mLastWarpedCorners[i].x - mWarpedCorners[i].x;
		const float dy = mLastWarpedCorners[i].y - mWarpedCorners[i].y;
		const float dd = dx*dx+dy*dy;
		if(dd > mMaxCornerDriftSq)
			return true;
	}
	return false;
}

void PatchWarper::updatePatch()
{
	//ProfileSection s("updatePatch");
	{
		//For some reason this is very slow in some machines when multi-threading.
		//cv::warpAffine(*mSourceImage, mPatch, mWarpMatrix, cv::Size(kPatchSize, kPatchSize), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);

		//Manual interpolation
		auto &srcImg = *mSourceImage;
		mPatch.create(kPatchSize, kPatchSize);
		
		//Check corners
		if (mWarpedCorners[0].x >= 0 && mWarpedCorners[0].y >= 0
			&& mWarpedCorners[0].x < srcImg.cols-1 && mWarpedCorners[0].y < srcImg.rows-1
			&& mWarpedCorners[1].x >= 0 && mWarpedCorners[1].y >= 0
			&& mWarpedCorners[1].x < srcImg.cols-1 && mWarpedCorners[1].y < srcImg.rows-1
			&& mWarpedCorners[2].x >= 0 && mWarpedCorners[2].y >= 0
			&& mWarpedCorners[2].x < srcImg.cols-1 && mWarpedCorners[2].y < srcImg.rows-1
			&& mWarpedCorners[3].x >= 0 && mWarpedCorners[3].y >= 0
			&& mWarpedCorners[3].x < srcImg.cols-1 && mWarpedCorners[3].y < srcImg.rows-1)
		{
			//Interpolate
			for (int v = 0; v < kPatchSize; ++v)
			{
				auto patchRow = mPatch[v];
				float m12yc = mWarpMatrix(0, 1)*v + mWarpMatrix(0, 2);
				float m22yc = mWarpMatrix(1, 1)*v + mWarpMatrix(1, 2);

				for (int u = 0; u < kPatchSize; ++u)
				{
					float srcX = mWarpMatrix(0, 0)*u + m12yc;
					float srcY = mWarpMatrix(1, 0)*u + m22yc;

					int srcXi = (int)srcX;
					int srcYi = (int)srcY;

					float alphaX = srcX - srcXi;
					float alphaY = srcY - srcYi;

					float rowMixLeft = srcImg(srcYi, srcXi)*(1.0f - alphaY) + srcImg(srcYi + 1, srcXi)*alphaY;
					float rowMixRight = srcImg(srcYi, srcXi + 1)*(1.0f - alphaY) + srcImg(srcYi + 1, srcXi + 1)*alphaY;

					patchRow[u] = (uchar)(rowMixLeft*(1.0f - alphaX) + rowMixRight*alphaX);
				}
			}
		}
	}
	mLastWarpMatrix = mWarpMatrix;
	for(int i=0; i<4; i++)
		mLastWarpedCorners[i] = mWarpedCorners[i];
}

} /* namespace dtslam */
