/*
 * ImagePyramid.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef IMAGEPYRAMID_H_
#define IMAGEPYRAMID_H_

#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace dtslam {

template<class T>
class ImagePyramid {
public:
	static int GetOctaveCount(int level0Width, int maxTopLevelWidth);

	int getOctaveCount() const {return (int)mOctaves.size();}

	cv::Mat_<T> &operator [](int octave) {return mOctaves[octave];}
	const cv::Mat_<T> &operator [](int octave) const {return mOctaves[octave];}

	cv::Mat_<T> &getTopLevel() {return mOctaves.back();}
	const cv::Mat_<T> &getTopLevel() const {return mOctaves.back();}

	void create(const cv::Mat_<T> &level0, int maxTopLevelWidth);

protected:
	std::vector<cv::Mat_<T>> mOctaves;
};

typedef ImagePyramid<uchar> ImagePyramid1b;
typedef ImagePyramid<cv::Vec2b> ImagePyramid2b;
typedef ImagePyramid<cv::Vec3b> ImagePyramid3b;

} /* namespace dtslam */

#endif /* IMAGEPYRAMID_H_ */
