/*
 * OpenCVDataSource.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef OPENCV_IMAGEDATASOURCE_H_
#define OPENCV_IMAGEDATASOURCE_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <GL/glew.h>
#include "ImageDataSource.h"

namespace dtslam
{

class OpenCVDataSource: public ImageDataSource
{
public:
    OpenCVDataSource(void);
    ~OpenCVDataSource(void);

    bool open(const std::string &videoFile);
    bool open(int deviceId);
    void close(void);

    void dropFrames(int count);
    bool update(void);

private:
    cv::VideoCapture mOpenCVCamera;

    bool finishOpen();
};

}

#endif /* VIDEOIMAGEDATASOURCE_H_ */
