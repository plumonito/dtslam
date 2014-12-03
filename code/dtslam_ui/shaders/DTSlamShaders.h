/*
 * DTSlamShaders.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef DTSlamShaders_H_
#define DTSlamShaders_H_

#include "ColorShader.h"
#include "TextShader.h"
#include "TextureShader.h"
#include "TextureWarpShader.h"
#include "TextRenderer.h"
#include "ColorCameraShader.h"
#include "StaticColors.h"

namespace dtslam
{

class DTSlamShaders {
public:
	DTSlamShaders();

	bool init()
	{
		bool res;
		res = mColor.init();
		res &= mTexture.init();
		res &= mTextureWarp.init();
		res &= mText.init();
		res &= mTextRenderer.init(&mText);
		res &= mColorCamera.init();
		return res;
	}

	void free()
	{
		mColor.free();
		mText.free();
		mTexture.free();
		mTextureWarp.free();
		mColorCamera.free();
	}

	ColorShader &getColor() {return mColor;}
	TextureShader &getTexture() {return mTexture;}
	TextureWarpShader &getTextureWarp() {return mTextureWarp;}
	TextRenderer &getText() {return mTextRenderer;}
	ColorCameraShader &getColorCamera() {return mColorCamera;}

protected:
	ColorShader mColor;
	TextureShader mTexture;
	TextureWarpShader mTextureWarp;
	TextShader mText;
    TextRenderer mTextRenderer;
    ColorCameraShader mColorCamera;
};

}

#endif /* DTSlamShaders_H_ */
