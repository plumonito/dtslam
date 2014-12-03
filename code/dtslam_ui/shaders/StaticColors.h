/*
 * StaticColors.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef STATICCOLORS_H_
#define STATICCOLORS_H_

#include <opencv2/core.hpp>

namespace dtslam {

class StaticColors
{
public:
	static cv::Vec4f White(float alpha=1.0f) {return cv::Vec4f(1,1,1,alpha);}
	static cv::Vec4f Black(float alpha=1.0f) {return cv::Vec4f(0,0,0,alpha);}
	static cv::Vec4f Gray(float alpha=1.0f,float brightness=0.5f) {return cv::Vec4f(brightness,brightness,brightness,alpha);}

	static cv::Vec4f Red(float alpha=1.0f, float brightness=1.0f) {return cv::Vec4f(brightness,0,0,alpha);}
	static cv::Vec4f Green(float alpha=1.0f, float brightness=1.0f) {return cv::Vec4f(0,brightness,0,alpha);}
	static cv::Vec4f Blue(float alpha=1.0f, float brightness=1.0f) {return cv::Vec4f(0,0,brightness,alpha);}

	static cv::Vec4f Yellow(float alpha=1.0f, float brightness=1.0f) {return cv::Vec4f(brightness,brightness,0,alpha);}
	static cv::Vec4f Purple(float alpha=1.0f, float brightness=1.0f) {return cv::Vec4f(brightness,0,brightness,alpha);}
	static cv::Vec4f Cyan(float alpha=1.0f, float brightness=1.0f) {return cv::Vec4f(0,brightness,brightness,alpha);}

	static cv::Vec4f ChangeAlpha(const cv::Vec4f &color, float alpha=1.0f) {return cv::Vec4f(color[0],color[1],color[2],alpha);}
private:
	StaticColors() {}
};

} /* namespace dtslam */

#endif /* STATICCOLORS_H_ */
