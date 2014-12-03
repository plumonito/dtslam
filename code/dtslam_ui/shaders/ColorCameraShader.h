/*
 * ColorCameraShader.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */
 
#ifndef COLOR_CAMERA_SHADER_H_
#define COLOR_CAMERA_SHADER_H_

#include <opencv2/core.hpp>
#include "dtslam/CameraModel.h"
#include "ShaderProgram.h"

namespace dtslam
{

class Pose3D;

class ColorCameraShader
{
public:
    ColorCameraShader() {}

    bool init();
    void free() {mProgram.free();}

    //mvp is in normal opencv row-major order
    void setMVPMatrix(const cv::Matx44f &mvp);
    void setCamera(const CameraModel_<RadialCameraDistortionModel> &camera);
    void setPose(const Pose3D &pose);

    void drawVertices(GLenum mode, const cv::Vec4f *vertices, int count, const cv::Vec4f &color);
    void drawVertices(GLenum mode, const cv::Vec4f *vertices, const cv::Vec4f *color, int count);
    void drawVertices(GLenum mode, const unsigned int *indices, unsigned int indexCount, const cv::Vec4f *vertices, const cv::Vec4f *color);

protected:
    ShaderProgram mProgram;
    int mUniformCameraK;
    int mUniformCameraDist;
    int mUniformCameraMaxRadiusSq;
    int mUniformRt;
    int mUniformMVPMatrix;
    int mAttribPosCoord;
    int mAttribColor;
};

}

#endif /* SHADERPROGRAM_H_ */
