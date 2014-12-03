/*
 * UserInterfaceInfo.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef USERINTERFACEINFO_H_
#define USERINTERFACEINFO_H_

#include <memory>
#include <cassert>
#include <opencv2/core.hpp>

namespace dtslam {

class UserInterfaceInfo {
public:
	static UserInterfaceInfo &Instance();

	const cv::Size2i &getScreenSize() const {return mScreenSize;}
	void setScreenSize(const cv::Size2i &sz) {mScreenSize=sz;}

	bool getKeyState(uchar key) const {assert(key<kKeyStateSize); return mKeyState[key];}
	void setKeyState(uchar key, bool state) {assert(key<kKeyStateSize); mKeyState[key] = state;}

	bool getSpecialKeyState(uchar key) const {assert(key<kKeyStateSize); return mSpecialKeyState[key];}
	void setSpecialKeyState(uchar key, bool state) {assert(key<kKeyStateSize); mSpecialKeyState[key] = state;}

protected:
	static std::unique_ptr<UserInterfaceInfo> gInstance;

	cv::Size2i mScreenSize;

	static const int kKeyStateSize=256;
	bool mKeyState[kKeyStateSize];
	bool mSpecialKeyState[kKeyStateSize];

	UserInterfaceInfo();
};

} /* namespace dtslam */

#endif /* USERINTERFACEINFO_H_ */
