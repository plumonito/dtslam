/*
 * ColorCameraShader.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */
#include "ColorCameraShader.h"
#include "dtslam/Pose3D.h"

namespace dtslam
{

bool ColorCameraShader::init()
{
    bool res = true;

	const char *uniforms[] = { "uCameraK","uCameraDist","uCameraMaxRadiusSq","uRt","uMVPMatrix" };
	int *uniformIds[] = {&mUniformCameraK, &mUniformCameraDist, &mUniformCameraMaxRadiusSq, &mUniformRt, &mUniformMVPMatrix};
	const int uniformsCount = sizeof(uniforms) / sizeof(uniforms[0]);

	const char *attribs[] =	{ "aPosCoord", "aColor" };
	int *attribIds[] = {&mAttribPosCoord, &mAttribColor };
	const int attribsCount = sizeof(attribs) / sizeof(attribs[0]);

	res &= mProgram.load("assets/colorCamera_render.vert", "assets/colorCamera_render.frag", uniforms, uniformIds, uniformsCount,
							 attribs, attribIds, attribsCount);
    return res;
}

void ColorCameraShader::setMVPMatrix(const cv::Matx44f &mvp)
{
	//Transpose to opengl column-major format
	cv::Matx44f mvpt = mvp.t();
    glUseProgram(mProgram.getId());
    glUniformMatrix4fv(mUniformMVPMatrix, 1, false, mvpt.val);
}
void ColorCameraShader::setCamera(const CameraModel_<RadialCameraDistortionModel> &camera)
{
	cv::Matx33f Kt = camera.getK().t();

	glUseProgram(mProgram.getId());
    glUniformMatrix3fv(mUniformCameraK, 1, false, Kt.val);
    glUniform2f(mUniformCameraDist, camera.getDistortionModel().getK1(), camera.getDistortionModel().getK2());
    glUniform1f(mUniformCameraMaxRadiusSq, camera.getDistortionModel().getMaxRadiusSq());
}

void ColorCameraShader::setPose(const Pose3D &pose)
{
	cv::Matx34f Rt = pose.getRt();
	cv::Matx44f Rtt;
	for(int j=0;j<3; ++j)
		for(int i=0;i<4; ++i)
			Rtt(i,j) = Rt(j,i);
	Rtt(0,3) = Rtt(1,3) = Rtt(2,3) = 0;
	Rtt(3,3) = 1;
	glUseProgram(mProgram.getId());
    glUniformMatrix4fv(mUniformRt, 1, false, Rtt.val);
}

void ColorCameraShader::drawVertices(GLenum mode, const cv::Vec4f *vertices, int count, const cv::Vec4f &color)
{
    std::vector<cv::Vec4f> colors;
    colors.resize(count, color);

    drawVertices(mode, vertices, colors.data(), count);
}

void ColorCameraShader::drawVertices(GLenum mode, const cv::Vec4f *vertices, const cv::Vec4f *color, int count)
{
    glUseProgram(mProgram.getId());

    glVertexAttribPointer(mAttribPosCoord, 4, GL_FLOAT, GL_FALSE, 0, vertices);
    glEnableVertexAttribArray(mAttribPosCoord);

    glVertexAttribPointer(mAttribColor, 4, GL_FLOAT, GL_FALSE, 0, color);
    glEnableVertexAttribArray(mAttribColor);

    glDrawArrays(mode, 0, count);

    glDisableVertexAttribArray(mAttribPosCoord);
    glDisableVertexAttribArray(mAttribColor);
}

void ColorCameraShader::drawVertices(GLenum mode, const unsigned int *indices, unsigned int indexCount, const cv::Vec4f *vertices, const cv::Vec4f *color)
{
    glUseProgram(mProgram.getId());

    glVertexAttribPointer(mAttribPosCoord, 4, GL_FLOAT, GL_FALSE, 0, vertices);
    glEnableVertexAttribArray(mAttribPosCoord);

    glVertexAttribPointer(mAttribColor, 4, GL_FLOAT, GL_FALSE, 0, color);
    glEnableVertexAttribArray(mAttribColor);

    glDrawElements(mode, indexCount, GL_UNSIGNED_INT, indices);

    glDisableVertexAttribArray(mAttribPosCoord);
    glDisableVertexAttribArray(mAttribColor);
}

}
