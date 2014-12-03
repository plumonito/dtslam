/*
 * WindowUtils.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef WINDOWUTILS_H_
#define WINDOWUTILS_H_

#include <vector>
#include <opencv2/core.hpp>
#include "dtslam/CameraModel.h"

namespace dtslam
{

class WindowUtils
{
public:
	static void BuildEpiLineVertices(const CameraModel &camera, const cv::Point3f &epipole, const cv::Point3f &infiniteXn, std::vector<cv::Point2f> &vertices);
};

} /* namespace dtslam */

#endif /* WINDOWUTILS_H_ */
