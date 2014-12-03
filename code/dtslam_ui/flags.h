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

#ifndef dtslam_FLAGS_UI_H_
#define dtslam_FLAGS_UI_H_

#include <gflags/gflags.h>

namespace dtslam
{
	DECLARE_int32(DriverCameraId);
	DECLARE_string(DriverDataPath);
	DECLARE_string(DriverVideoFile);
	DECLARE_string(DriverSequenceFormat);
	DECLARE_int32(DriverSequenceStartIdx);
	DECLARE_int32(DriverDropFrames);
	DECLARE_int32(DriverMaxImageWidth);
	DECLARE_bool(DriverSingleThreaded);
	DECLARE_string(DriverRecordPath);
	
	DECLARE_double(MapDrawScale);
}

#endif 
