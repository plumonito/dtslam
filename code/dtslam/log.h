/*
* log.h
*
* Copyright(C) 2014, University of Oulu, all rights reserved.
* Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
* Third party copyrights are property of their respective owners.
* Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
*          Kihwan Kim(kihwank@nvidia.com)
* Author : Daniel Herrera C.
*/

#ifndef DTSLAM_LOG_H_
#define DTSLAM_LOG_H_

#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <memory>

// The define below enables logging!
// Note: each log call takes a lock on a mutex due to multi-threading. Thus, logging considerably reduces performance!
// 		 We probably should switch to glog.
#define ENABLE_LOG

#ifdef ENABLE_LOG
	#define DTSLAM_LOG (dtslam::Log(__FILE__, __LINE__, __FUNCTION__))
#else
	#define DTSLAM_LOG (dtslam::NullLog(__FILE__, __LINE__, __FUNCTION__))
#endif

namespace cv
{
	template <class T>
	class Size_;

	template <class T>
	class Point_;

	template <class T>
	class Mat_;

	typedef Mat_<unsigned char> Mat1b;
	typedef Mat_<float> Mat1f;

	template <class T, int M, int N>
	class Matx;
}

namespace dtslam
{

class MatlabDataLog
{
public:
	static MatlabDataLog &Instance()
	{
		if(!gInstance)
		{
			gInstance.reset(new MatlabDataLog());
			gInstance->mStream.open(gInstance->mLogPath + "/" + gInstance->mLogFilename);
		}

		return *gInstance;
	}
	static std::ofstream &Stream()
	{
		return Instance().mStream;
	}

	static void ClearVar(const std::string &varName)
	{
		Stream() << varName << "=[];\n";
		Instance().mVariables.insert(varName);
	}

	template<class T>
	static void SetValue(const std::string &varName, const T &value)
	{
		Instance().mVariables.insert(varName);
		Stream() << varName << "=[" << value << "];\n" << std::flush;
	}

	template<class T>
	static void AddValue(const std::string &varName, const T &value)
	{
		if(Instance().mVariables.find(varName)==Instance().mVariables.end())
			ClearVar(varName);
		Stream() << varName << "=[" << varName << "," << value << "];\n" << std::flush;
	}

	template<class T>
	static void AddCell(const std::string &varName, const T &value)
	{
		auto it = Instance().mCellArrays.insert(std::make_pair(varName, 1));
		if(it.second)
			Stream() << varName << "={};\n";
		else
			it.first->second++;

		Stream() << varName << "{" << it.first->second << "}=[" << value << "];\n";
	}

	static void AddCell(const std::string &varName)
	{
		AddCell(varName, "");
	}

	template<class T>
	static void AddValueToCell(const std::string &varName, const T &value)
	{
		auto it = Instance().mCellArrays.find(varName);
		if(it==Instance().mCellArrays.end())
			AddCell(varName, value);
		else
			Stream() << varName << "{" << it->second << "}=[" << varName << "{" << it->second << "}," << value << "];\n" << std::flush;
	}

private:
	static std::unique_ptr<MatlabDataLog> gInstance;

	std::string mLogPath;
	std::string mLogFilename;
    std::ofstream mStream;

    std::unordered_set<std::string> mVariables;

    std::unordered_map<std::string, int> mCellArrays;

    MatlabDataLog():
    	mLogPath("C:/code/dslam/code/matlab"),
    	mLogFilename("vars.m")
    {}
};

class Log
{
public:
	Log(const std::string &file, int line, const std::string &function);
	Log(std::ostream &stream);
	Log(): Log(std::cout) {}

	~Log();

	template<class T>
	Log &operator <<(const T &value)
	{
		mStream << value;
		return *this;
	}

	Log &operator <<(std::ostream & (*value)(std::ostream&))
	{
		mStream << value;
		return *this;
	}

private:
	static std::mutex gMutex;
	std::lock_guard<std::mutex> mLock;
	std::ostream mStream;
};

class NullLog
{
public:
	NullLog(const std::string &file, int line, const std::string &function) 
	{}

	template<class T>
	NullLog &operator <<(const T &value) 
	{ 
		return *this; 
	}
};


template<class T>
std::ostream &operator <<(std::ostream &s, const std::vector<T> &value)
{
	auto it=value.begin();
	s << "[";
	if(it!=value.end())
	{
		s << *it;
		for(++it; it!=value.end(); ++it)
			s << "," << *it;
	}
	s << "]";
	return s;
}

}


/////////////////////////////////////////////////////////////////////////////////////
// Template implementations
/////////////////////////////////////////////////////////////////////////////////////
#include <opencv2/core.hpp>

namespace dtslam
{

//template<class T, int M, int N>
//std::ostream &operator <<(std::ostream &s, const cv::Matx<T,M,N> &value)
//{
//	s << "[";
//	for(int j=0; j<value.rows; j++)
//	{
//		const T *row = &value(j,0);
//		s << (T)row[0];
//		for(int i=1; i<value.cols; i++)
//		{
//			s << "," << (T)row[i];
//		}
//		if(j<value.rows-1)
//			s << "; ";
//	}
//	s << "]";
//	return s;
//}

//template<class T>
//std::ostream &operator <<(std::ostream &s, const cv::Mat_<T> &value)
//{
//	s << "[";
//	for(int j=0; j<value.rows; j++)
//	{
//		const T *row = value[j];
//		s << (T)row[0];
//		for(int i=1; i<value.cols; i++)
//		{
//			s << "," << (T)row[i];
//		}
//		if(j<value.rows-1)
//			s << ";\n";
//	}
//	s << "]";
//	return s;
//}

}

#endif
