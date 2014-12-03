/*
 * stdutils.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef STDUTILS_H_
#define STDUTILS_H_

namespace dtslam
{

class stdutils
{
public:
	template<class T>
	static bool IsTaskRunning(std::future<T> &future);
};

/////////////////////////////////////////////////////////////////////////////////////////
// Implementation
template<class T>
bool stdutils::IsTaskRunning(std::future<T> &future)
{
	if(!future.valid())
		return false;

	auto status = future.wait_for(std::chrono::seconds(0));
	if(status == std::future_status::ready)
		return false;
	else
		return true;
}

}


#endif /* STDUTILS_H_ */
