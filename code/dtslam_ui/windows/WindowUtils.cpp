/*
 * WindowUtils.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "WindowUtils.h"

namespace dtslam
{

void WindowUtils::BuildEpiLineVertices(const CameraModel &camera, const cv::Point3f &epipole, const cv::Point3f &infiniteXn, std::vector<cv::Point2f> &vertices)
{
	const int kDivisions = 100;

	const cv::Point3f dx = (infiniteXn-epipole)*(1.0f/kDivisions);

	cv::Point3f vertexXn = epipole;
	cv::Point2f vertex;

	if(fabs(vertexXn.z) > 0.0001f)
	{
		vertex = camera.projectFromWorld(vertexXn);
		vertices.push_back(vertex);
	}

	for(int k=0; k<kDivisions; ++k)
	{
		vertexXn+=dx;
		vertex = camera.projectFromWorld(vertexXn);
		vertices.push_back(vertex);
	}
}

} /* namespace dtslam */
