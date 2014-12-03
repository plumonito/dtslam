/*
 * CameraModel.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "CameraModel.h"
#include "CameraModelCeres.h"

namespace dtslam {

template class CameraModel_<NullCameraDistortionModel>;
template class CameraModel_<RadialCameraDistortionModel>;

} /* namespace dtslam */
