/*
 * Pose3D.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "Pose3D.h"

#include <ceres/rotation.h>

namespace dtslam {

void FullPose3D::setFromArray(const std::vector<double> &array)
{
	assert(array.size()==6);

	cv::Matx33d rd;

	ceres::AngleAxisToRotationMatrix(array.data(), ceres::RowMajorAdapter3x3(rd.val));
	mRotation = rd;
	mTranslation[0] = (float)array[3];
	mTranslation[1] = (float)array[4];
	mTranslation[2] = (float)array[5];
}

void FullPose3D::copyToArray(std::vector<double> &array) const
{
	assert(array.size()==6);

	cv::Matx33d rd;

	rd = mRotation;
	ceres::RotationMatrixToAngleAxis(ceres::RowMajorAdapter3x3((const double *)rd.val),array.data());
	array[3] = mTranslation[0];
	array[4] = mTranslation[1];
	array[5] = mTranslation[2];
}

void FullPose3D::serialize(Serializer &s, cv::FileStorage &fs) const
{
	fs << "R" << mRotation;
	fs << "t" << mTranslation;
}
void FullPose3D::deserialize(Deserializer &s, const cv::FileNode &node)
{
	node["R"] >> mRotation;
	node["t"] >> mTranslation;
}

void RelativeRotationPose3D::setFromArray(const std::vector<double> &array)
{
	assert(array.size()==3);

	cv::Matx33d rd;

	ceres::AngleAxisToRotationMatrix(array.data(), ceres::RowMajorAdapter3x3(rd.val));
	mRelativeRotation = rd;
}

void RelativeRotationPose3D::copyToArray(std::vector<double> &array) const
{
	assert(array.size()==3);

	cv::Matx33d rd;

	rd = mRelativeRotation;
	ceres::RotationMatrixToAngleAxis(ceres::RowMajorAdapter3x3((const double *)rd.val),array.data());
}

void RelativeRotationPose3D::serialize(Serializer &s, cv::FileStorage &fs) const
{
	fs << "base" << s.addObject(mBasePose);
	fs << "relR" << mRelativeRotation;
}
void RelativeRotationPose3D::deserialize(Deserializer &s, const cv::FileNode &node)
{
	mBasePose = s.getObject<Pose3D>(node["base"]);
	node["relR"] >> mRelativeRotation;
}

void RelativePose3D::setFromAbsolute(const cv::Matx33f &rotation, const cv::Vec3f &translation)
{
	cv::Matx33f baseRT = mBasePose->getRotation().t();
	mRelativePose.setRotation(rotation*baseRT);
	mRelativePose.setTranslation(translation - mRelativePose.getRotationRef()*mBasePose->getTranslation());
}

void RelativePose3D::setFromArray(const std::vector<double> &array)
{
	mRelativePose.setFromArray(array);
}

void RelativePose3D::copyToArray(std::vector<double> &array) const
{
	mRelativePose.copyToArray(array);
}

void RelativePose3D::serialize(Serializer &s, cv::FileStorage &fs) const
{
	fs << "base" << s.addObject(mBasePose);
	fs << "relR" << mRelativePose.getRotationRef();
	fs << "relt" << mRelativePose.getTranslationRef();
}
void RelativePose3D::deserialize(Deserializer &s, const cv::FileNode &node)
{
	mBasePose = s.getObject<Pose3D>(node["base"]);

	node["relR"] >> mRelativePose.getRotationRef();
	node["relt"] >> mRelativePose.getTranslationRef();
}

} /* namespace dtslam */
