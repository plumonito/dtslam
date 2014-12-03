/*
 * ShaderProgram.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Dawid Pajak
 */

#ifndef SHADERPROGRAM_H_
#define SHADERPROGRAM_H_

#include <map>
#include <string>
#include <GL/glew.h>
#include "dtslam/log.h"

namespace dtslam
{

class ShaderProgram
{
public:
    ShaderProgram();
    ~ShaderProgram();

    static bool CreateFragmentProgramFromStrings(GLuint programId, GLuint vertexShaderId, GLuint fragmentShaderId,
                                                 const char *vertexShader, const char *fragmentShader);
    static bool CreateFragmentProgram(GLuint &destProgramId, const char *vertexShader, const char *fragmentShader);
    static bool LoadFragmentProgram(GLuint &programId, const char *vertexShaderFileName,
                                        const char *fragmentShaderFileName);

    bool load(const char *vertexShaderFilename, const char *fragmentShaderFilename,
    		const char *uniformNames[], int *uniformIds[], int uniformCount,
    		const char *attribNames[], int *attribIds[], int attribCount);
    void free();

    bool isLoaded() const { return mLoaded; }
    unsigned int getId() const {return mId; }

protected:
    bool mLoaded;
    unsigned int mId;
};

}

#endif /* SHADERPROGRAM_H_ */
