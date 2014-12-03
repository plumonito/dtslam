/*
 * CameraModelCeres.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef CAMERAMODELCERES_H_
#define CAMERAMODELCERES_H_

#include "CameraModel.h"
#include <ceres/jet.h>

namespace dtslam
{

template<class TDistortionModel>
void CameraModel_<TDistortionModel>::projectFromWorldJacobian(const cv::Point3f &xc, cv::Vec3f &ujac, cv::Vec3f &vjac) const
{
	ceres::Jet<float, 3> xJet(xc.x, 0);
	ceres::Jet<float, 3> yJet(xc.y, 1);
	ceres::Jet<float, 3> zJet(xc.z, 2);

	ceres::Jet<float, 3> uJet;
	ceres::Jet<float, 3> vJet;

	projectFromWorld(xJet,yJet,zJet,uJet,vJet);
	ujac[0] = uJet.v[0];
	ujac[1] = uJet.v[1];
	ujac[2] = uJet.v[2];

	vjac[0] = vJet.v[0];
	vjac[1] = vJet.v[1];
	vjac[2] = vJet.v[2];
}

}

#endif /* CAMERAMODELCERES_H_ */
