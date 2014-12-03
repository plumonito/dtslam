/*
 * ReprojectionError3D.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "ReprojectionError3D.h"

namespace dtslam
{

void ReprojectionError3D::evalToErrors(const cv::Matx33f &R, const cv::Vec3f &t, const cv::Vec3f &x, const float errorThreshold, MatchReprojectionErrors &errors) const
{
	std::vector<float> allResiduals(2*getImagePointCount());

	computeAllResidualsRmat(R.val, t.val, x.val, allResiduals.data());
	residualsToErrors(allResiduals, errorThreshold, errors);
}

void ReprojectionError3D::evalToErrors(const cv::Vec3d &Rparams, const cv::Vec3d &t, const cv::Vec3d &x, const float errorThreshold, MatchReprojectionErrors &errors) const
{
	std::vector<double> allResiduals(2*getImagePointCount());

	computeAllResidualsRparams(Rparams.val, t.val, x.val, allResiduals.data());
	residualsToErrors(allResiduals, errorThreshold, errors);
}

}
