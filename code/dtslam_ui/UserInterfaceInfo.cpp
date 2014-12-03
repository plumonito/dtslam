/*
 * UserInterfaceInfo.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "UserInterfaceInfo.h"

namespace dtslam {

std::unique_ptr<UserInterfaceInfo> UserInterfaceInfo::gInstance;

UserInterfaceInfo &UserInterfaceInfo::Instance()
{
	if(gInstance.get() == NULL)
		gInstance.reset(new UserInterfaceInfo());
	return *gInstance;
}

UserInterfaceInfo::UserInterfaceInfo()
{
	memset(mKeyState, 0, sizeof(mKeyState));
	memset(mSpecialKeyState, 0, sizeof(mSpecialKeyState));
}

} /* namespace dtslam */
