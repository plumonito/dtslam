/*
 * ShaderProgram.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Dawid Pajak
 */

#include "ShaderProgram.h"
#include <memory>
#include <fstream>
#include "dtslam/log.h"

namespace dtslam
{

ShaderProgram::ShaderProgram()
        : mLoaded(false)
{
}

ShaderProgram::~ShaderProgram()
{
    if (mLoaded)
    {
        //DTSLAM_LOG << "Program not freed before destruction!\n";
    }
}

static bool CheckShaderStatus(GLuint shader)
{
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == 0)
    {
        int messageLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &messageLength);
        if (messageLength > 1)
        {
            std::unique_ptr<char[]> message(new char[messageLength]);
            glGetShaderInfoLog(shader, messageLength, 0, message.get());
            DTSLAM_LOG << "GLSL compiler message: " << message.get() << "\n";
        }
        return false;
    }

    return true;
}

static bool CheckFragmentProgramStatus(GLuint program, GLenum mode)
{
    GLint status;
    glGetProgramiv(program, mode, &status);
    if (status == 0)
    {
        int messageLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &messageLength);
        if (messageLength > 1)
        {
            std::unique_ptr<char[]> message(new char[messageLength]);
            glGetProgramInfoLog(program, messageLength, 0, message.get());
            DTSLAM_LOG << "GLSL compiler message: " << message.get() << "\n";
        }
        return false;
    }

    return true;
}

bool ShaderProgram::CreateFragmentProgramFromStrings(GLuint programId, GLuint vertexShaderId, GLuint fragmentShaderId,
                                             const char *vertexShader, const char *fragmentShader)
{
    glShaderSource(vertexShaderId, 1, &vertexShader, 0);
    glCompileShader(vertexShaderId);
    if (!CheckShaderStatus(vertexShaderId))
    {
        return false;
    }

    glShaderSource(fragmentShaderId, 1, &fragmentShader, 0);
    glCompileShader(fragmentShaderId);
    if (!CheckShaderStatus(fragmentShaderId))
    {
        return false;
    }

    // link the fragment program
    glAttachShader(programId, vertexShaderId);
    glAttachShader(programId, fragmentShaderId);
    glLinkProgram(programId);
    if (!CheckFragmentProgramStatus(programId, GL_LINK_STATUS))
    {
        return false;
    }

    glValidateProgram(programId);
    if (!CheckFragmentProgramStatus(programId, GL_VALIDATE_STATUS))
    {
        return false;
    }

    return true;
}

bool ShaderProgram::CreateFragmentProgram(GLuint &destProgramId, const char *vertexShader, const char *fragmentShader)
{
    GLuint vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint programId = glCreateProgram();

    if (!CreateFragmentProgramFromStrings(programId, vertexShaderId, fragmentShaderId, vertexShader, fragmentShader))
    {
        glDeleteProgram(programId);
        glDeleteShader(fragmentShaderId);
        glDeleteShader(vertexShaderId);
        return false;
    }

    glDeleteShader(fragmentShaderId);
    glDeleteShader(vertexShaderId);

    destProgramId = programId;
    return true;
}

bool ShaderProgram::LoadFragmentProgram(GLuint &programId, const char *vertexShaderFileName,
                                    const char *fragmentShaderFileName)
{
	std::ifstream vertexShaderFile, fragmentShaderFile;

	vertexShaderFile.open(vertexShaderFileName);
    if (!vertexShaderFile)
    {
        DTSLAM_LOG << "unable to open \"" << vertexShaderFileName << "\"!\n";
        return false;
    }

    std::stringstream vertexStr;
    vertexStr << vertexShaderFile.rdbuf();
    vertexShaderFile.close();

    fragmentShaderFile.open(fragmentShaderFileName);
    if (!fragmentShaderFile)
    {
        DTSLAM_LOG << "unable to open \"" << fragmentShaderFileName << "\"!\n";
        return false;
    }

    std::stringstream fragmentStr;
    fragmentStr << fragmentShaderFile.rdbuf();
    fragmentShaderFile.close();
    fragmentShaderFile.close();

    return CreateFragmentProgram(programId, vertexStr.str().c_str(), fragmentStr.str().c_str());
}

bool ShaderProgram::load(const char *vertexShaderFilename, const char *fragmentShaderFilename,
		const char *uniformNames[], int *uniformIds[], int uniformCount,
		const char *attribNames[], int *attribIds[], int attribCount)
{
    //Texture shader
    mLoaded = LoadFragmentProgram(mId, vertexShaderFilename, fragmentShaderFilename);
    if (!mLoaded)
    {
        DTSLAM_LOG << "Error loading " << vertexShaderFilename << " or " << fragmentShaderFilename << "\n";
        return false;
    }

    for (int i = 0; i < uniformCount; i++)
    {
        *uniformIds[i] = glGetUniformLocation(mId, uniformNames[i]);
    }
    for (int i = 0; i < attribCount; i++)
    {
        *attribIds[i] = glGetAttribLocation(mId, attribNames[i]);
    }

    return true;
}

void ShaderProgram::free()
{
	if(mLoaded)
	{
		glDeleteProgram(mId);
		mLoaded = false;
	}
}

}
