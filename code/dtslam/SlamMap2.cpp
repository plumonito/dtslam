/*
 * SlamMap2.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "SlamMap.h"

//Note: This file exists only because VisualC++ was crashing in SlamMap.cpp. Splitting it into two removed the issue.

namespace dtslam {

SlamFeatureMeasurement *SlamFeature::getBestMeasurementForMatching(const cv::Vec3f &poseCenter) const
{
	SlamFeatureMeasurement *bestM = NULL;
	float minDistSq = std::numeric_limits<float>::infinity();

	for (int i = 0; i < (int)mMeasurements.size(); ++i)
	{
		auto &m = *mMeasurements[i];
		if (m.getPositions().size() > 1)
			continue;

		float distSq = cvutils::PointDistSq(poseCenter, m.getKeyFrame().getPose().getCenter());
		if (distSq < minDistSq)
		{
			minDistSq = distSq;
			bestM = &m;
		}
	}
	return bestM;
}

} /* namespace dtslam */
