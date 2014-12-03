/*
 * TextureHelper.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef TEXTUREHELPER_H_
#define TEXTUREHELPER_H_

#include <opencv2/core/core.hpp>

#undef ERROR //From miniglog
#include <GL/glew.h>
#include <GL/freeglut.h>
#undef ERROR //From miniglog

namespace dtslam
{

class TextureHelper
{
public:
    TextureHelper();
    virtual ~TextureHelper();

    void create(int format, const cv::Size &size);
    void free();
    void update(const cv::Mat &img);

    bool isValid() const {return mIsValid;}
    uint getTarget() const {return GL_TEXTURE_2D;}
    uint getId() const {return mId;}
    const cv::Size &getSize() const {return mSize;}

protected:
    bool mIsValid;
    uint mId;
    cv::Size mSize;
    int mFormat;

    void updateRGBA(const cv::Mat &img);
    void updateRGB(const cv::Mat &img);
    void updateSingleChannel(const cv::Mat &img);

    void updateInternal(const cv::Mat &img);
};

}

#endif /* TEXTUREHELPER_H_ */
