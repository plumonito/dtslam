/*
 * PatchWarper.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef PATCHWARPER_H_
#define PATCHWARPER_H_

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include "CameraModel.h"
#include "Pose3D.h"

namespace dtslam {

///////////////////////////////////
// PatchWarper
class PatchWarper {
public:
	static const int kPatchSizeBits=3;
	static const int kPatchSize=(1<<kPatchSizeBits); //1<<3==8
	static const int kPatchCenterOffset=3;
	static const int kPatchRightSize = kPatchSize-kPatchCenterOffset;

	PatchWarper();

	static cv::Mat ExtractPatch(const cv::Mat &imageSrc, const cv::Point2f &centerSrc, const int octave=0, const int centerOffset=kPatchCenterOffset, const int size=kPatchSize);
	static cv::Mat1b ExtractPatch(const cv::Mat1b &imageSrc, const cv::Point2f &centerSrc, const cv::Point2f &centerDst, const int octave, const cv::Matx33f &homography, const int centerOffset=kPatchCenterOffset, const int size=kPatchSize);

	const cv::Mat1b &getPatch() const {return mPatch;}

	void setSource(const cv::Mat1b *srcImage, const cv::Point2f &srcPosition);
	void setSource(const CameraModel *srcCamera, const cv::Mat1b *srcImage, int srcOctave, const cv::Point2f &srcPosition, const cv::Point3f &srcXn);
	const cv::Point2f &getSourcePosition() const {return mSourcePosition;}

	void calculateWarp(const cv::Point2f &srcRightCenter, const cv::Point2f &srcCenterBottom);
	void calculateWarp(const cv::Matx33f &hDst2Src, const cv::Point2i &dstPosition);
	void calculateWarp(const cv::Matx33f &dst2srcR, const CameraModel &dstCamera, const cv::Point2f &dstCenter, const cv::Point2f &dstRight, const cv::Point2f &dstBottom);
	void calculateWarp(const cv::Matx33f &dst2srcR, const CameraModel &dstCamera, const cv::Point2f &dstCenter, const int dstScale);
	void calculateWarp(const cv::Matx33f &dst2srcR, const CameraModel &dstCamera, const cv::Matx23f &dst2cameraAffine, const cv::Point2f &dstCenter, const int dstScale);

	void calculateWarp3D(const Pose3D &srcPose, const cv::Vec3f &planeNormal, const cv::Vec3f &planePoint, const Pose3D &dstPose, const CameraModel &dstCamera, const cv::Matx23f &dst2imageffine, const cv::Point2f &imgCenter, const int imgScale);
	cv::Point2f warpPoint3D(const Pose3D &srcPose, const cv::Vec3f &planeNormal, const cv::Vec3f &planePoint, const Pose3D &dstPose, const CameraModel &dstCamera, const cv::Point2f &dstPoint);

	void warpCorners(const cv::Matx23f &warp, cv::Point2f warpedCorners[3]);

	bool patchNeedsUpdate();

	void updatePatch();

private:
	float mMaxCornerDriftSq;

	const CameraModel *mSourceCamera;
	const cv::Mat1b *mSourceImage;
	int mSourceScale;
	cv::Point2f mSourcePosition;
	cv::Point3f mSourceXn;

	cv::Matx23f mLastWarpMatrix;
	cv::Point2f mLastWarpedCorners[4];

	cv::Matx23f mWarpMatrix;
	cv::Point2f mWarpedCorners[4];

	cv::Mat1b mPatch;
	//Eigen::Matrix<float, 8, 8, Eigen::AutoAlign | Eigen::RowMajor> mp;
};

class SlamFeature;
class PatchWarperCache
{
public:
	PatchWarper &get(SlamFeature *feature)
	{
		auto it = mFeatures.emplace(feature, feature);
		return it.first->second.getWarper();
	}

protected:
	class Entry
	{
	public:
		Entry(SlamFeature *feature):
			mFeature(feature)//, mLastTimeUsed(0)
		{}

		SlamFeature * getFeature() const {return mFeature;}
		PatchWarper &getWarper() {return mWarper;}

	protected:
		SlamFeature *mFeature;
		PatchWarper mWarper;
		//double mLastTimeUsed; //This can be used to determine when to drop this feature from the cache
	};

	std::unordered_map<SlamFeature *, Entry> mFeatures;
};

} /* namespace dtslam */

#endif /* PATCHWARPER_H_ */
