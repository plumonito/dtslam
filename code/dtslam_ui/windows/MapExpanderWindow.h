/*
 * MapExpanderWindow.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef MAPEXPANDERWINDOW_H_
#define MAPEXPANDERWINDOW_H_

#include "BaseWindow.h"

namespace dtslam
{

class SlamMapExpander;

class MapExpanderWindow: public BaseWindow
{
public:
	MapExpanderWindow():
		BaseWindow("MapExpanderWindow")
	{}

	bool init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize);
	void showHelp() const;

	void resize();
    void updateState();
    void draw();

protected:
    ViewportTiler mTiler;
    TextureHelper mFeatureCoverageTexture;
	cv::Mat3b mFeatureCoverageImage;

    int mOctaveCount;

    struct MatchData
    {
    	int octave;
    	cv::Point2f position;
    	cv::Vec4f color;
    	float angle;
    };
    std::vector<MatchData> mMatchesToDraw;

    SlamMapExpander *mMapExpander;
};

} /* namespace dtslam */

#endif /* MAPEXPANDERWINDOW_H_ */
