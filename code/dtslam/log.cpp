/*
 * log.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

 #include "log.h"
#include <iostream>
#include <opencv2/core.hpp>

namespace dtslam
{
	std::unique_ptr<MatlabDataLog> MatlabDataLog::gInstance;

	std::mutex Log::gMutex;

	Log::Log(const std::string &file, int line, const std::string &function):
			Log()
	{
	}

	Log::Log(std::ostream &stream):
			mLock(gMutex),
			mStream(stream.rdbuf())
	{
	}

	Log::~Log()
	{
		mStream.flush();
	}
}
