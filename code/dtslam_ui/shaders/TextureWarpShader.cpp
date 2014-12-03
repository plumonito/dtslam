/*
 * TextureWarpShader.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */
 
#include "TextureWarpShader.h"
#include <cassert>

namespace dtslam
{

bool TextureWarpShader::init()
{
	bool res;

	{
		const char *uniforms[] = { "uMVPMatrix", "uTexture" ,"uHomography"};
		int *uniformIds[] = {&mUniformMVPMatrix, &mUniformTexture, &mUniformHomographyMatrix};
		const int uniformsCount = sizeof(uniforms) / sizeof(uniforms[0]);

		const char *attribs[] = { "aPosCoord", "aTexCoord" };
		int *attribIds[] = {&mAttribPosCoord, &mAttribTexCoord};
		const int attribsCount = sizeof(attribs) / sizeof(attribs[0]);

		res = mProgram.load("assets/texture_warp.vert", "assets/texture_warp.frag", uniforms, uniformIds, uniformsCount, attribs, attribIds,
								attribsCount);
	}
    {
		const char *uniforms[] = { "uMVPMatrix", "uTexture" ,"uHomography","uAlpha"};
		int *uniformIds[] = {&mUniformMVPMatrix_Alpha, &mUniformTexture_Alpha, &mUniformHomographyMatrix_Alpha, &mUniformAlpha_Alpha};
		const int uniformsCount = sizeof(uniforms) / sizeof(uniforms[0]);

		const char *attribs[] = { "aPosCoord", "aTexCoord" };
		int *attribIds[] = {&mAttribPosCoord_Alpha, &mAttribTexCoord_Alpha};
		const int attribsCount = sizeof(attribs) / sizeof(attribs[0]);

		res &= mProgram_Alpha.load("assets/texture_warp.vert", "assets/texture_warp_fixed_alpha.frag", uniforms, uniformIds, uniformsCount, attribs, attribIds,
								attribsCount);
    }

    return res;
}

void TextureWarpShader::setMVPMatrix(const cv::Matx44f &mvp)
{
	//Transpose to opengl column-major format
	cv::Matx44f mvpt = mvp.t();
    glUseProgram(mProgram.getId());
    glUniformMatrix4fv(mUniformMVPMatrix, 1, false, mvpt.val);

    glUseProgram(mProgram_Alpha.getId());
    glUniformMatrix4fv(mUniformMVPMatrix_Alpha, 1, false, mvpt.val);
}

void TextureWarpShader::renderTexture(GLuint target, GLuint id, const cv::Matx33f &homography, const cv::Size &imageSize,
                                    const cv::Point2f &screenOrigin)
{
    const float kDepth = 1.0f;
    cv::Vec4f const vertices[] =
    { cv::Vec4f(screenOrigin.x + (float)imageSize.width - 1, screenOrigin.y, kDepth, 1.0f), cv::Vec4f(screenOrigin.x,
                                                                                                      screenOrigin.y,
                                                                                                      kDepth, 1.0f),
      cv::Vec4f(screenOrigin.x + (float)imageSize.width - 1, screenOrigin.y + (float)imageSize.height - 1, kDepth,
                1.0f),
      cv::Vec4f(screenOrigin.x, screenOrigin.y + (float)imageSize.height - 1, kDepth, 1.0f) };
    cv::Vec2f const textureCoords[] =
    { cv::Vec2f(1, 0), cv::Vec2f(0, 0), cv::Vec2f(1, 1), cv::Vec2f(0, 1) };
    renderTexture(GL_TRIANGLE_STRIP, target, id, homography, imageSize, vertices, textureCoords, 4);
}

void TextureWarpShader::renderTexture(GLenum mode, GLuint target, GLuint id, const cv::Matx33f &homography, const cv::Size2i &imageSize, const cv::Vec4f *vertices,
                                    const cv::Vec2f *textureCoords, int count)
{
    assert(target == GL_TEXTURE_2D);

    glUseProgram(mProgram.getId());

    cv::Matx33f finalH;
    cv::Matx33f normHR = cv::Matx33f::eye();
	normHR(0, 0) = (float)(imageSize.width - 1);
	normHR(1, 1) = (float)(imageSize.height - 1);
    cv::Matx33f normHL = cv::Matx33f::eye();
    normHL(0, 0) = 1.0f / (imageSize.width - 1);
    normHL(1, 1) = 1.0f / (imageSize.height - 1);

    cv::Matx33f finalH_t = (normHL * homography * normHR).t();

    glUniformMatrix3fv(mUniformHomographyMatrix, 1, false, finalH_t.val);

    // setup uniforms
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(target, id);
    glUniform1i(mUniformTexture, 0);

    // drawing quad
    glVertexAttribPointer(mAttribPosCoord, 4, GL_FLOAT, GL_FALSE, 0, vertices);
    glVertexAttribPointer(mAttribTexCoord, 2, GL_FLOAT, GL_FALSE, 0, textureCoords);
    glEnableVertexAttribArray(mAttribPosCoord);
    glEnableVertexAttribArray(mAttribTexCoord);
    glDrawArrays(mode, 0, count);
    glDisableVertexAttribArray(mAttribPosCoord);
    glDisableVertexAttribArray(mAttribTexCoord);
}


void TextureWarpShader::renderTexture(GLuint target, GLuint id, const cv::Matx33f &homography, float alpha, const cv::Size &imageSize,
                                    const cv::Point2f &screenOrigin)
{
    const float kDepth = 1.0f;
    cv::Vec4f const vertices[] =
    { cv::Vec4f(screenOrigin.x + (float)imageSize.width - 1, screenOrigin.y, kDepth, 1.0f), cv::Vec4f(screenOrigin.x,
                                                                                                      screenOrigin.y,
                                                                                                      kDepth, 1.0f),
      cv::Vec4f(screenOrigin.x + (float)imageSize.width - 1, screenOrigin.y + (float)imageSize.height - 1, kDepth,
                1.0f),
      cv::Vec4f(screenOrigin.x, screenOrigin.y + (float)imageSize.height - 1, kDepth, 1.0f) };
    cv::Vec2f const textureCoords[] =
    { cv::Vec2f(1, 0), cv::Vec2f(0, 0), cv::Vec2f(1, 1), cv::Vec2f(0, 1) };
    renderTexture(GL_TRIANGLE_STRIP, target, id, homography, alpha, imageSize, vertices, textureCoords, 4);
}

void TextureWarpShader::renderTexture(GLenum mode, GLuint target, GLuint id, const cv::Matx33f &homography, float alpha, const cv::Size2i &imageSize, const cv::Vec4f *vertices,
                                    const cv::Vec2f *textureCoords, int count)
{
    assert(target == GL_TEXTURE_2D);

    glUseProgram(mProgram_Alpha.getId());

    cv::Matx33f finalH;
    cv::Matx33f normHR = cv::Matx33f::eye();
	normHR(0, 0) = (float)(imageSize.width - 1);
	normHR(1, 1) = (float)(imageSize.height - 1);
    cv::Matx33f normHL = cv::Matx33f::eye();
    normHL(0, 0) = 1.0f / (imageSize.width - 1);
    normHL(1, 1) = 1.0f / (imageSize.height - 1);

    cv::Matx33f finalH_t = (normHL * homography * normHR).t();

    glUniformMatrix3fv(mUniformHomographyMatrix_Alpha, 1, false, finalH_t.val);

    glUniform1f(mUniformAlpha_Alpha, alpha);

    // setup uniforms
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(target, id);
    glUniform1i(mUniformTexture_Alpha, 0);

    // drawing quad
    glVertexAttribPointer(mAttribPosCoord_Alpha, 4, GL_FLOAT, GL_FALSE, 0, vertices);
    glVertexAttribPointer(mAttribTexCoord_Alpha, 2, GL_FLOAT, GL_FALSE, 0, textureCoords);
    glEnableVertexAttribArray(mAttribPosCoord_Alpha);
    glEnableVertexAttribArray(mAttribTexCoord_Alpha);
    glDrawArrays(mode, 0, count);
    glDisableVertexAttribArray(mAttribPosCoord_Alpha);
    glDisableVertexAttribArray(mAttribTexCoord_Alpha);
}

}
