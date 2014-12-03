/*
 * TextShader.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

 #ifndef TEXT_SHADER_H_
#define TEXT_SHADER_H_

#include <opencv2/core.hpp>
#include "ShaderProgram.h"

namespace dtslam
{

class TextShader
{
public:
	TextShader() {}

    bool init();
    void free() {mProgram.free();}

    //mvp is in normal opencv row-major order
    void setMVPMatrix(const cv::Matx44f &mvp);

    void renderText(GLenum mode, GLuint textureId, const cv::Vec4f *vertices, const cv::Vec2f *textureCoords, int count, const cv::Vec4f &color);

protected:
    ShaderProgram mProgram;
    int mUniformMVPMatrix;
    int mUniformTexture;
    int mUniformColor;
    int mAttribPosCoord;
    int mAttribTexCoord;
};

}

#endif /* SHADERPROGRAM_H_ */
