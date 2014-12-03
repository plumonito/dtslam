/*
 * TextureWarpShader.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */
 
#ifndef TEXTURE_WARP_SHADER_H_
#define TEXTURE_WARP_SHADER_H_

#include <opencv2/core.hpp>
#include "ShaderProgram.h"

namespace dtslam
{

class TextureWarpShader
{
public:
	TextureWarpShader() {}

    bool init();
    void free() {mProgram.free(); mProgram_Alpha.free();}

    //mvp is in normal opencv row-major order
    void setMVPMatrix(const cv::Matx44f &mvp);

    void renderTexture(GLuint target, GLuint id, const cv::Matx33f &homography, const cv::Size &imageSize) {renderTexture(target,id,homography,imageSize,cv::Point2f(0,0));}
    void renderTexture(GLuint target, GLuint id, const cv::Matx33f &homography, const cv::Size &imageSize,
                                        const cv::Point2f &screenOrigin);
    void renderTexture(GLenum mode, GLuint target, GLuint id, const cv::Matx33f &homography, const cv::Size2i &imageSize, const cv::Vec4f *vertices,
                                        const cv::Vec2f *textureCoords, int count);

    void renderTexture(GLuint target, GLuint id, const cv::Matx33f &homography, float alpha, const cv::Size &imageSize) {renderTexture(target,id,homography,alpha,imageSize,cv::Point2f(0,0));}
    void renderTexture(GLuint target, GLuint id, const cv::Matx33f &homography, float alpha, const cv::Size &imageSize,
                                        const cv::Point2f &screenOrigin);
    void renderTexture(GLenum mode, GLuint target, GLuint id, const cv::Matx33f &homography, float alpha, const cv::Size2i &imageSize, const cv::Vec4f *vertices,
                                        const cv::Vec2f *textureCoords, int count);

protected:
    ShaderProgram mProgram;
    int mUniformMVPMatrix;
    int mUniformHomographyMatrix;
    int mUniformTexture;
    int mAttribPosCoord;
    int mAttribTexCoord;

    ShaderProgram mProgram_Alpha;
    int mUniformMVPMatrix_Alpha;
    int mUniformHomographyMatrix_Alpha;
    int mUniformTexture_Alpha;
    int mUniformAlpha_Alpha;
    int mAttribPosCoord_Alpha;
    int mAttribTexCoord_Alpha;
};

}

#endif
