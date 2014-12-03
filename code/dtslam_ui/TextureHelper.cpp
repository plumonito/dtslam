/*
 * TextureHelper.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "TextureHelper.h"
#include <cassert>
#include <opencv2/imgproc/imgproc.hpp>
#include <GL/glew.h>
#include "dtslam/log.h"

namespace dtslam
{

//Support for this is added with GL_EXT_unpack_subimage
#define GL_UNPACK_ROW_LENGTH                0x0CF2

TextureHelper::TextureHelper()
{
    mIsValid = false;
    mId = 0;
}

TextureHelper::~TextureHelper()
{
	if (mIsValid)
	{
		//DTSLAM_LOG << "Texture not freed before destruction!\n";
	}
}

void TextureHelper::create(int format, const cv::Size &size)
{
    free();

    assert(format == GL_RGBA || format == GL_RGB || format == GL_LUMINANCE || format == GL_ALPHA);

    mFormat = format;

    glGenTextures(1, &mId);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mId);
    glTexImage2D(GL_TEXTURE_2D, 0, mFormat, size.width, size.height, 0, mFormat, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    mSize = size;
    mIsValid = true;
}

void TextureHelper::free()
{
    if(mIsValid)
    {
        // Delete the texture
        glDeleteTextures(1, &mId);
        mIsValid = false;
    }
}

void TextureHelper::update(const cv::Mat &img)
{
    glBindTexture(GL_TEXTURE_2D, mId);

    switch(mFormat)
    {
    case GL_RGBA: updateRGBA(img); break;
    case GL_RGB: updateRGB(img); break;
    case GL_ALPHA:
    case GL_LUMINANCE: updateSingleChannel(img); break;
    }
}

void TextureHelper::updateRGBA(const cv::Mat &img)
{
    assert(img.channels() == 4);

    switch(img.channels())
    {
    case 4: //Input is RGBA
        if(!img.isContinuous())
        {
            glPixelStorei(GL_UNPACK_ROW_LENGTH, img.step[0] / 4);
        }
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img.cols, img.rows, GL_RGBA, GL_UNSIGNED_BYTE, img.data);
        if(!img.isContinuous())
        {
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        }
        break;
    }
}

void TextureHelper::updateRGB(const cv::Mat &img)
{
    assert(img.channels() == 3);

    switch(img.channels())
    {
    case 3: //Input is RGB
        if(!img.isContinuous())
        {
            glPixelStorei(GL_UNPACK_ROW_LENGTH, img.step[0] / 3);
        }
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img.cols, img.rows, GL_RGB, GL_UNSIGNED_BYTE, img.data);
        if(!img.isContinuous())
        {
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        }
        break;
    }
}

void TextureHelper::updateSingleChannel(const cv::Mat &img)
{
    assert(img.channels() == 1);

    switch(img.channels())
    {
    case 1: //Input is one channel
        if(!img.isContinuous())
        {
            glPixelStorei(GL_UNPACK_ROW_LENGTH, img.step[0]);
        }
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img.cols, img.rows, mFormat, GL_UNSIGNED_BYTE, img.data);
        if(!img.isContinuous())
        {
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        }
        break;
    }
}

/*
void TextureHelper::update(const cv::Mat &img, uchar alpha)
{
    cv::Mat4b temp;
    temp.create(img.rows, img.cols);

    if (img.channels() == 1)
    {
        cv::Mat srcMats[] =
        { img, cv::Mat1b(img.rows, img.cols, alpha) };
        int fromTo[] =
        { 0, 0, 0, 1, 0, 2, 1, 3 };

        cv::mixChannels(srcMats, 2, &temp, 1, fromTo, 4);
    }
    else if (img.channels() == 3)
    {
        cv::Mat srcMats[] =
        { img, cv::Mat1b(img.rows, img.cols, alpha) };
        int fromTo[] =
        { 0, 0, 1, 1, 2, 2, 3, 3 };

        cv::mixChannels(srcMats, 2, &temp, 1, fromTo, 4);
    }
    else
    {
        LOG("ERROR: too many channels in texture.");
        return;
    }

    updateInternal(temp);
}
*/

void TextureHelper::updateInternal(const cv::Mat &img)
{
    glBindTexture(GL_TEXTURE_2D, mId);

    // Update the texture
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if(img.isContinuous())
    {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img.cols, img.rows, GL_RGBA, GL_UNSIGNED_BYTE, img.data);
    }
    else
    {
        DTSLAM_LOG << "Warning: image not continous, wasting time repacking texture.\n";
        uchar *data = new uchar[img.cols * img.rows * 4];
        uchar *src = img.data;
        uchar *dst = data;
        for(int j = 0; j < img.rows; j++)
        {
            for(int i = 0; i < img.cols * 4; i++)
            {
                *(dst++) = src[i];
            }
            src += img.step[0];
        }
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img.cols, img.rows, GL_RGBA, GL_UNSIGNED_BYTE, data);
        delete data;
    }
}

}
