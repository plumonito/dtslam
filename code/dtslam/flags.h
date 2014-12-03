/*
 * flags.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef dtslam_FLAGS_H_
#define dtslam_FLAGS_H_

#undef GFLAGS_DLL_DECLARE_FLAG
#define GFLAGS_DLL_DECLARE_FLAG
#include <gflags/gflags.h>

namespace dtslam
{
	DECLARE_double(CameraFx);
	DECLARE_double(CameraFy);
	DECLARE_double(CameraU0);
	DECLARE_double(CameraV0);
	DECLARE_int32(CameraWidth);
	DECLARE_int32(CameraHeight);

	DECLARE_double(CameraK1);
	DECLARE_double(CameraK2);


	DECLARE_int32(PyramidMaxTopLevelWidth);
	DECLARE_int32(SBIMaxWidth);
	DECLARE_int32(FASTThreshold);
	DECLARE_int32(FrameKeypointGridSize);


	DECLARE_double(MinDepth);
	DECLARE_int32(FramesFor2DTracking);


	DECLARE_int32(TrackerMaxFeatures);
	DECLARE_int32(TrackerMaxFeaturesPerOctave);
	DECLARE_int32(TrackerMinMatchCount);

	DECLARE_int32(TrackerSelectFeaturesGridSize);
	DECLARE_int32(MatcherPixelSearchDistance);
	DECLARE_int32(TrackerOutlierPixelThreshold);


	DECLARE_int32(MatcherNonMaximaPixelSize);
	DECLARE_int32(MatcherMaxZssdScore);
	DECLARE_double(MatcherBestScorePercentThreshold);
	DECLARE_double(WarperMaxCornerDrift);


	DECLARE_double(ExpanderMinTriangulationAngle);
	DECLARE_int32(ExpanderMinNewTriangulationsForKeyFrame);
	DECLARE_double(ExpanderNewCoverageRatioForKeyFrame);


	DECLARE_int32(PoseMinRansacIterations);
	DECLARE_int32(PoseMaxRansacIterations);
	DECLARE_int32(PoseMinTrackLength);
	DECLARE_double(PoseStableRatioThreshold);

	DECLARE_bool(PoseUse2D);
	DECLARE_bool(DisableBA);
	DECLARE_int32(GlobalBAFrameCount);
	DECLARE_bool(SavePose);
	DECLARE_bool(DisableRegions);
}

#endif 
