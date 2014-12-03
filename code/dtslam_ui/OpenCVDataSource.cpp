/*
 * OpenCVDataSource.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "OpenCVDataSource.h"
#include <opencv2/imgproc.hpp>
#include "dtslam/Profiler.h"
#include "dtslam/log.h"

namespace dtslam
{

OpenCVDataSource::OpenCVDataSource(void)
{
}

OpenCVDataSource::~OpenCVDataSource(void)
{
    close();
}

bool OpenCVDataSource::open(const std::string &videoFile)
{
    if (mOpenCVCamera.isOpened())
    {
        // already opened
        return false;
    }

    if (!mOpenCVCamera.open(videoFile))
    {
        return false;
    }

    return finishOpen();
}

bool OpenCVDataSource::open(int deviceId)
{
    if (mOpenCVCamera.isOpened())
    {
        // already opened
        return false;
    }

    if (!mOpenCVCamera.open(deviceId))
    {
        return false;
    }

    dropFrames(5);
    return finishOpen();
}
 bool OpenCVDataSource::finishOpen()
{
    int videoWidth = (int)mOpenCVCamera.get(cv::CAP_PROP_FRAME_WIDTH);
	int videoHeight = (int)mOpenCVCamera.get(cv::CAP_PROP_FRAME_HEIGHT);

    setSourceSize(cv::Size(videoWidth, videoHeight));

    return true;
}

void OpenCVDataSource::close(void)
{
	releaseGl();
    if (mOpenCVCamera.isOpened())
    {
        mOpenCVCamera.release();
    }
}

void OpenCVDataSource::dropFrames(int count)
{
	for(int i=0; i<count; ++i)
	{
	    cv::Mat frame;
	    mOpenCVCamera.read(frame);
	}
}

bool OpenCVDataSource::update(void)
{
    cv::Mat frame;
    if(!mOpenCVCamera.read(frame))
    {
    	return false;
    }

    assert(frame.channels()==3);
	int frameId = (int)mOpenCVCamera.get(cv::CAP_PROP_POS_FRAMES);

    // convert to RGBY format
    ImageDataSource::update(cv::Mat3b(frame), frameId);

    return true;
}

}
