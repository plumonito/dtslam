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
	DEFINE_int32(DriverCameraId, -1, "Id of the camera to open (OpenCV). 0 is the first camera.");
	DEFINE_string(DriverDataPath, "assets", "Path to all data files (videos and image sequences)");
	DEFINE_string(DriverVideoFile, "translation.mp4", "Name of the video file to use (e.g. translation.mp4). This is appended to the data path. If both VideoFile and SequenceFormat are empty, the camera is used.");
	DEFINE_string(DriverSequenceFormat, "", "sprintf format for the sequence (e.g. \"/cityOfSights/CS_BirdsView_L0/Frame_%.5d.jpg\". This is appended to the data path. If both VideoFile and SequenceFormat are empty, the camera is used.");
	DEFINE_int32(DriverSequenceStartIdx, 0, "Start index for the image sequence.");
	DEFINE_int32(DriverDropFrames, 0, "The system will ignore this many frames per iteration, effectively lowering the frame rate or skipping frames in a video.");
	DEFINE_int32(DriverMaxImageWidth, 960, "Maximum width of input image. Input will be downsampled to be under this width.");
	DEFINE_bool(DriverSingleThreaded, false, "Use a single thread for easier debugging.");
	DEFINE_string(DriverRecordPath, "record/", "Path where the frames will be stored in case of recording.");

	DEFINE_double(MapDrawScale, 1.0, "Scale to draw the map in the MapWindow.");
}

