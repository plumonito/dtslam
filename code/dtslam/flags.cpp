/*
 * flags.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include <gflags/gflags.h>

namespace dtslam
{
	DEFINE_double(CameraFx, 1759.9583, "Camera focal length along X.");
	DEFINE_double(CameraFy, 1758.2239, "Camera focal length along Y.");
	DEFINE_double(CameraU0, 963.80029, "Camera principal point X.");
	DEFINE_double(CameraV0, 541.04811, "Camera principal point Y.");
	DEFINE_int32(CameraWidth, 1920, "Camera image width.");
	DEFINE_int32(CameraHeight, 1080, "Camera image height.");

	DEFINE_double(CameraK1, 0.16262, "Camera radial distortion coefficient 1.");
	DEFINE_double(CameraK2, 0.67445, "Camera radial distortion coefficient 2.");


	DEFINE_int32(PyramidMaxTopLevelWidth, 240, "Maximum width of the highest pyramid level for a frame.");
	DEFINE_int32(SBIMaxWidth, 60, "Maximum width for the Small Blurry Image, input will be downsampled until width is less than this.");
	DEFINE_int32(FASTThreshold, 10, "Threshold for the FAST keypoint detector");
	DEFINE_int32(FrameKeypointGridSize, 30, "The grid size used to sort keypoints in the frame");


	DEFINE_int32(FramesFor2DTracking, 5, "Max number of frames to use as a source of 2D features during tracking. Nearest frames will be used first.");
	DEFINE_double(MinDepth, 1.0, "Minimum valid depth for a feature.");


	DEFINE_int32(TrackerMaxFeatures, 400, "Max number of features to search for in all levels.");
	DEFINE_int32(TrackerMaxFeaturesPerOctave, 500, "Max number of features to search for in each levels.");
	DEFINE_int32(TrackerMinMatchCount, 20, "Minimum number of matches that must be found in order to compute a pose. If less are found tracking is aborted.");

	DEFINE_int32(TrackerSelectFeaturesGridSize, 30, "Grid size for the select feature indexer.");
	DEFINE_int32(MatcherPixelSearchDistance, 8, "The search distance for matching features (distance from point projection or from epiplar line). Units in pixels of the highest pyramid level.");
	DEFINE_int32(TrackerOutlierPixelThreshold, 3, "Threshold for determining a match an outlier (in pixels).");


	DEFINE_int32(MatcherNonMaximaPixelSize, 5, "Size of the non-maxima suppresion area to reject matching keypoints that are too close.");
	DEFINE_int32(MatcherMaxZssdScore, 12800, "The max zssd score to consider a feature matched.");
	DEFINE_double(MatcherBestScorePercentThreshold, 1.3, "Secondary match score must be below (MatcherBestScorePercentThreshold*bestScore) to be accepted as a match.");
	DEFINE_double(WarperMaxCornerDrift, 2, "Maximum offset of the project corners before the patch has to be recalculated.");


	DEFINE_double(ExpanderMinTriangulationAngle, 2, "Minimum angle (in degrees) for a 2D feature to be triangulated.");
	DEFINE_int32(ExpanderMinNewTriangulationsForKeyFrame, 100, "Minimum number of 2D features that are ready for triangulation that will trigger a new key frame.");
	DEFINE_double(ExpanderNewCoverageRatioForKeyFrame, 0.3, "Minimum ratio of cells covered by new features over cells covered by old features that will trigger a new key frame.");


	DEFINE_int32(PoseMinRansacIterations, 10, "Minimum number of RANSAC iterations to run for estimating pose.");
	DEFINE_int32(PoseMaxRansacIterations, 50, "Maximum number of RANSAC iterations to run for estimating pose.");
	DEFINE_int32(PoseMinTrackLength, -1, "Minimum track length required to consider it stable.");
	DEFINE_double(PoseStableRatioThreshold, 0.1, "When the difference between stable essential and rotation inliers is this percentage of the essential inliers, the motion is no longer considered a pure rotation.");

	DEFINE_bool(PoseUse2D, true, "Enable the use of 2D features for pose estimation, if false only 3D features will be used.");
	DEFINE_bool(DisableBA, false, "Disable the background BA task.");
	DEFINE_int32(GlobalBAFrameCount, 100000, "Max number of frames to bundle adjust during global BA.");
	DEFINE_bool(SavePose, false, "Saves the tracker pose after each frame. Use for evaluating against ground truth.");
	DEFINE_bool(DisableRegions, true, "Disables the creation of new regions. When regions are disabled and only 2D features are observed the system will force the pose estimate to be a pure rotation.");
}

