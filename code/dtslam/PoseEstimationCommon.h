/*
* PoseEstimationCommon.h
*
* Copyright(C) 2014, University of Oulu, all rights reserved.
* Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
* Third party copyrights are property of their respective owners.
* Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
*          Kihwan Kim(kihwank@nvidia.com)
* Author : Daniel Herrera C.
*/


#ifndef POSEESTIMATIONCOMMON_H_
#define POSEESTIMATIONCOMMON_H_

#include <vector>
#include "flags.h"

namespace dtslam {

struct MatchReprojectionErrors
{
	MatchReprojectionErrors():
		isInlier(false)
	{}

	bool isInlier;
	float bestReprojectionErrorSq;
	std::vector<float> reprojectionErrorsSq; //One for each point in match.imagePoints
	std::vector<bool> isImagePointInlier;
};

}

#endif
