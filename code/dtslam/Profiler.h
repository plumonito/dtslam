/*
 * Profiler.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef PROFILER_H_
#define PROFILER_H_

#include <memory>
#include <map>
#include <vector>
#include <stack>
#include <string>
#include <iomanip>
#include <thread>
#include <mutex>
#include <chrono>
#include <ostream>
#include <cassert>

// The define below enables the profiler!
#define ENABLE_PROFILER

namespace dtslam
{

class ProfilerSectionData
{
public:
	inline ProfilerSectionData(ProfilerSectionData *parent, const std::string &key);

	ProfilerSectionData *getParent() const {return mParent;}
	const std::string &getKey() const {return mKey;}
	double getLastTic() const {return mLastTic;}

	inline double getTime() const;
	int getSampleCount() const {return mSampleCount;}

	void setLastTic(double val) {mLastTic=val;}
	inline void addSample(double tics);

	const std::map<std::string, std::unique_ptr<ProfilerSectionData>> &getSubsections() const {return mChildSections;}

	inline ProfilerSectionData *getSubsection(const std::string &subkey);

	inline void reset();

    template<typename T>
    inline T &logStats(T &stream, const std::string &prefix, const double parentTime);

protected:
	ProfilerSectionData *mParent;
	std::string mKey;
	double mLastTic;
	double mTotalTics;
	int mSampleCount;
	std::map<std::string, std::unique_ptr<ProfilerSectionData>> mChildSections;
};

class ProfilerThreadData
{
public:
	ProfilerThreadData(std::thread::id id);

	const std::thread::id &getId() const {return mId;}
	void setId(const std::thread::id &newId) {mId = newId;}

	const std::string &getName() const {return mName;}
	void setName(const std::string &name) {mName = name;}

	ProfilerSectionData *getRootSection() {return &mRootSection;}

    inline void tic(const std::string &sectionKey, bool isTopLevelSection);
    inline void toc();

    template<typename T>
    inline T &logStats(T &stream);
    inline void reset();

protected:
    std::thread::id mId;
	std::string mName;
    ProfilerSectionData mRootSection;
    std::stack<ProfilerSectionData *> mActiveSections;
};

/**
 * @brief Class to keep track of runtimes for different sections of code (i.e. runtime profiling).
 * Can be used directly with tic(string) and toc(string), or through the ProfileSection() class. All
 * profiling code is removed if ENABLE_PROFILER is not defined.
 */
class Profiler
{
public:
    inline static Profiler &Instance();

    inline double now();
    inline void tic(const std::string &sectionKey, bool isTopLevelSection);
    inline void toc();

    bool getShowTotals() const {return mShowTotals;}
    void setShowTotals(bool value) {mShowTotals = value;}

//    void logStats();
    template<typename T>
    inline T &logStats(T &stream);
    inline void reset();

    void setCurrentThreadName(const std::string &name);
    ProfilerThreadData *getThreadData();

protected:
    std::mutex mMutex;
    std::map<std::thread::id, std::unique_ptr<ProfilerThreadData>> mThreads;
    bool mShowTotals;

protected:
    /**
     * @brief Singleton instance
     */
    static std::unique_ptr<Profiler> gInstance;

    /**
     * @brief Singleton class, hidden constructor.
     */
    Profiler();
};

template<typename T>
T &operator <<(T &stream, Profiler &profiler)
{
    return profiler.logStats(stream);
}

/**
 * @brief Used to profile a section by just constructing and destructing this object.
 */
class ProfileSection
{
public:
    ProfileSection(const std::string &sectionKey, bool isTopLevel)
    {
#ifdef ENABLE_PROFILER
        Profiler::Instance().tic(sectionKey, isTopLevel);
#endif
    }
    ProfileSection(const std::string &sectionKey)
            : ProfileSection(sectionKey, false)
    {
    }

    ~ProfileSection()
    {
#ifdef ENABLE_PROFILER
        Profiler::Instance().toc();
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
ProfilerSectionData::ProfilerSectionData(ProfilerSectionData *parent, const std::string &key):
		mParent(parent), mKey(key), mLastTic(0), mTotalTics(0), mSampleCount(0)
{
}

double ProfilerSectionData::getTime() const
{
	if(mSampleCount>0)
		return mTotalTics/mSampleCount;
	else
		return 0;
}

void ProfilerSectionData::addSample(double tics)
{
	mTotalTics+=tics;
	mSampleCount++;
}

ProfilerSectionData *ProfilerSectionData::getSubsection(const std::string &subkey)
{
	auto itSection = mChildSections.find(subkey);
	if(itSection==mChildSections.end())
	{
		ProfilerSectionData *data;
		data = new ProfilerSectionData(this, subkey);
		mChildSections.insert(std::make_pair(subkey, std::unique_ptr<ProfilerSectionData>(data)));
		return data;
	}
	else
		return itSection->second.get();
}

void ProfilerSectionData::reset()
{
	mTotalTics = 0;
	mSampleCount = 0;
	for(auto it=mChildSections.begin(); it!=mChildSections.end(); it++)
		it->second->reset();
}

template<typename T>
T &ProfilerSectionData::logStats(T &stream, const std::string &prefix, const double parentTime)
{
	const bool showTotal = Profiler::Instance().getShowTotals();

	if(showTotal)
	{
		int time = static_cast<int>(mTotalTics);
		stream << prefix << mKey << ": " << time << "ms";
		if(parentTime > 0)
		{
			int percent = static_cast<int>(100*time/parentTime);
			stream << " (" << percent << "%)";
		}
		stream << "\n";

		int totalChildrenTime = 0;
		std::string newPrefix = prefix + "-";
		for(auto it=mChildSections.begin(); it!=mChildSections.end(); it++)
		{
			it->second->logStats(stream, newPrefix, time);
			totalChildrenTime += static_cast<int>(it->second->mTotalTics);
		}

		if(time > 0 && mChildSections.size() > 1)
		{
			int otherTime = time-totalChildrenTime;
			int otherPercent = 100*otherTime/time;
			if(otherPercent > 0)
				stream << newPrefix << "other: " << otherTime << "ms (" << otherPercent << "%)\n";
		}
	}
	else
	{
		double time = getTime();
		stream << prefix << mKey << ": " << std::fixed << std::setprecision(2) << time << "ms";
		stream << " x " << mSampleCount;
		stream << "\n";

		std::string newPrefix = prefix + "-";
		for(auto it=mChildSections.begin(); it!=mChildSections.end(); it++)
			it->second->logStats(stream, newPrefix, time);
	}

	return stream;
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerThreadData::tic(const std::string &sectionKey, bool isTopLevelSection)
{
	double now = Profiler::Instance().now();

	ProfilerSectionData *data;
	if(isTopLevelSection)
		data = mRootSection.getSubsection(sectionKey);
	else
		data = mActiveSections.top()->getSubsection(sectionKey);

	data->setLastTic(now);
	mActiveSections.push(data);
}

void ProfilerThreadData::toc()
{
	//Get toc time
	double now(Profiler::Instance().now());

	//Get tic time
	assert(mActiveSections.size() > 1);
	double duration = now - mActiveSections.top()->getLastTic();
	mActiveSections.top()->addSample(duration);

	mActiveSections.pop();
}

template<typename T>
T &ProfilerThreadData::logStats(T &stream)
{
	stream << "Thread ";
	if(mName.empty())
		stream << mId;
	else
		stream << mName;
	stream << ":" << std::endl;
	for(auto it=mRootSection.getSubsections().begin(); it!=mRootSection.getSubsections().end(); it++)
		it->second->logStats(stream, "-", 0);
	return stream;
}

void ProfilerThreadData::reset()
{
	//This clears timing but preserves the keys
	mRootSection.reset();
}

///////////////////////////////////////////////////////////////////////////////////////
double Profiler::now(void)
{
	//TODO: we now use std::chrono. This could be more elegant. No need to cast to double. We could store a time_point inside the Profiler structures.
    return (double)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

Profiler &Profiler::Instance()
{
	if (!gInstance.get())
	{
		gInstance.reset(new Profiler());
	}
	return *gInstance;
}

void Profiler::tic(const std::string &sectionKey, bool isTopLevelSection)
{
	ProfilerThreadData *thread = getThreadData();
	thread->tic(sectionKey, isTopLevelSection);
}

void Profiler::toc()
{
	ProfilerThreadData *thread = getThreadData();
	thread->toc();
}

template<typename T>
T &Profiler::logStats(T &stream)
{
#ifdef ENABLE_PROFILER
	std::vector<ProfilerThreadData*> allData;
	{
		std::lock_guard<std::mutex> lock(mMutex);

		for(auto it=mThreads.begin(); it!=mThreads.end(); ++it)
			allData.push_back(it->second.get());
	}

	//stream << "Profiler stats:" << std::endl;
	for(auto it=allData.begin(); it!=allData.end(); ++it)
		(*it)->logStats(stream);
#endif
	return stream;
}

void Profiler::reset()
{
	std::lock_guard<std::mutex> lock(mMutex);

	for(auto it=mThreads.begin(); it!=mThreads.end(); ++it)
		it->second->reset();
}

}

#endif /* PROFILER_H_ */
