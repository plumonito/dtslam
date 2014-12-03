/*
 * ImageDataSource.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "ImageDataSource.h"
#include <opencv2/imgproc.hpp>
#include "dtslam/cvutils.h"

namespace dtslam
{

void ImageDataSource::setSourceSize(const cv::Size &sz)
{
	mSourceSize = sz;
	createBuffers();
}

void ImageDataSource::createBuffers()
{
	int scale=1<<mDownsampleCount;
	cv::Size dsz(mSourceSize.width/scale, mSourceSize.height/scale);
	mImgGray.create(dsz);
	mImgColor.create(dsz);

	releaseGl();

    glGenTextures(1, &mSourceTextureId);
    glBindTexture(GL_TEXTURE_2D, mSourceTextureId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mSourceSize.width, mSourceSize.height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
}

void ImageDataSource::releaseGl()
{
	if (mSourceTextureId != 0)
    {
        glDeleteTextures(1, &mSourceTextureId);
        mSourceTextureId = 0;
    }

}

void ImageDataSource::update(const cv::Mat3b &source, double captureTime)
{
	mCaptureTime = captureTime;

	//Switch channels around
	cv::Mat3b sourceRgb;
	cv::cvtColor(source, sourceRgb, cv::COLOR_BGR2RGB);

    // upload preview data to a texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mSourceTextureId);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sourceRgb.cols, sourceRgb.rows, GL_RGB,
                    GL_UNSIGNED_BYTE, sourceRgb.data);

    //Downsample color image
    mImgColor = cv::Mat3b();
    cvutils::DownsampleImage(sourceRgb, mImgColor, mDownsampleCount);

    // convert to gray format
    mImgGray = cv::Mat1b();
    cv::cvtColor(mImgColor, mImgGray, cv::COLOR_RGB2GRAY);
}

}
