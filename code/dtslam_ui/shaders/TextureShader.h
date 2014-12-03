/*
 * TextureShader.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */
#ifndef TEXTURE_SHADER_H_
#define TEXTURE_SHADER_H_

#include <opencv2/core.hpp>
#include "ShaderProgram.h"

namespace dtslam
{

class TextureShader
{
public:
	TextureShader() {}

    bool init();
    void free() {mProgram.free(); mProgramFA.free();}

    //mvp is in normal opencv row-major order
    void setMVPMatrix(const cv::Matx44f &mvp);

    void renderTexture(GLuint target, GLuint id, const cv::Size &imageSize) {renderTexture(target,id,imageSize,cv::Point2f(0,0));}
    void renderTexture(GLuint target, GLuint id, const cv::Size &imageSize,
                                        const cv::Point2f &screenOrigin);
    void renderTexture(GLenum mode, GLuint target, GLuint id, const cv::Vec4f *vertices,
                                        const cv::Vec2f *textureCoords, int count);

    void renderTexture(GLuint target, GLuint id, const cv::Size &imageSize, float alpha) {renderTexture(target,id,imageSize,cv::Point2f(0,0), alpha);}
    void renderTexture(GLuint target, GLuint id, const cv::Size &imageSize,
                                        const cv::Point2f &screenOrigin, float alpha);
    void renderTexture(GLenum mode, GLuint target, GLuint id, const cv::Vec4f *vertices,
                                        const cv::Vec2f *textureCoords, int count, float alpha);

protected:
    ShaderProgram mProgram;
    int mUniformMVPMatrix;
    int mUniformTexture;
    int mAttribPosCoord;
    int mAttribTexCoord;

    ShaderProgram mProgramFA;
    int mUniformMVPMatrixFA;
    int mUniformTextureFA;
    int mUniformAlphaFA;
    int mAttribPosCoordFA;
    int mAttribTexCoordFA;
};

}

#endif
