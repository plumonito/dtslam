/*
 * TextRenderer.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef TEXTRENDERER_H_
#define TEXTRENDERER_H_

#include <opencv2/core/core.hpp>
#include <sstream>
#include <memory>
#include "../TextureHelper.h"
#include "TextShader.h"

namespace dtslam
{

class TextRenderer;

class TextRendererStream
{
public:
	TextRendererStream(TextRenderer &renderer): mRenderer(renderer)
	{
	}

	~TextRendererStream();

	void flush();

	void setColor(const cv::Vec4f &color);

    template<class T>
    TextRendererStream &operator <<(const T &value);
	
protected:
    TextRenderer &mRenderer;
    std::stringstream mStream;
};

class TextRenderer
{
public:
    TextRenderer();
    ~TextRenderer();

    /**
     * @brief Builds the texture used for text rendering.
     */
    bool init(TextShader *shader);

    void setActiveFontSmall() {mActiveFont = &mFontData[0];}
    void setActiveFontBig() {mActiveFont = &mFontData[1];}

    void setMVPMatrix(const cv::Matx44f &mvp) {mShader->setMVPMatrix(mvp);}
    void setRenderCharHeight(float height) {mRenderCharHeight = height;}
    void setCaret(const cv::Vec4f &caret) {mRenderCaret = mRenderCaret0 = caret;}
    void setCaret(const cv::Point2f &caret) {setCaret(cv::Vec4f(caret.x, caret.y, 1, 1));}
    void setColor(const cv::Vec4f &color) {mActiveColor = color;}

    void renderText(const std::string &str)
    {
        std::stringstream ss(str);
        renderText(ss);
    }
    void renderText(std::stringstream &str);

protected:
    class TextFontData
    {
    public:
        static const int kCharCount = 255;

        TextureHelper mTexture;
        float mCharAspect[kCharCount];
        float mCharTexStart[kCharCount+1];
        float mCharTexEnd[kCharCount+1];
    };

    TextShader *mShader;

    std::vector<TextFontData> mFontData;
    TextFontData *mActiveFont;
    cv::Vec4f mActiveColor;

    float mRenderCharHeight;

    cv::Vec4f mRenderCaret0;
    cv::Vec4f mRenderCaret;

    void prepareFontData(TextFontData &data, int face, double scale, int thickness, int lineType);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
template<class T>
TextRendererStream &TextRendererStream::operator <<(const T &value)
{
	mStream << value;
	return *this;
}

}

#endif /* TEXTRENDERER_H_ */
