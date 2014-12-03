/*
 * Serializer.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef SERIALIZER_H_
#define SERIALIZER_H_

#include <opencv2/core.hpp>
#include <stdint.h>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <exception>
#include <cassert>
#include "ProjectConfig.h"

#if POINTER_SIZE == 8
namespace cv
{

static inline FileStorage &operator<<(FileStorage& fs, intptr_t value)
{
    std::ostringstream ss;
    ss << value;

    return fs << ss.str();
}

static inline void operator >> (const FileNode& n, intptr_t& value)
{
    std::string str;
    n >> str;

    char *endptr = NULL;
    value = strtoll(str.c_str(), &endptr, 10);

    // XXX: how to report errors??
    assert(*endptr == '\0');
}

}
#endif


namespace dtslam {

class Pose3D;
class FullPose3D;
class RelativeRotationPose3D;
class RelativePose3D;

class SlamKeyFrame;
class SlamMap;
class SlamRegion;
class Slam2DSection;
class SlamFeature;

class Serializer;
class Deserializer;

class ISerializable
{
public:
	virtual ~ISerializable() {}
	virtual const std::string getTypeName() const=0;
	virtual void serialize(Serializer &s, cv::FileStorage &fs) const=0;
	virtual void deserialize(Deserializer &s, const cv::FileNode &node)=0;
};

class Serializer
{
public:
	//Path can be empty, else it must include the path separator. Absolute filename path=path+filename.
	//Images will be stored in the same path but with filenames according to the prefix.
	void open(const std::string &path, const std::string &filename)
	{
		mPath = path;
		mFilename = filename;
		mFS.open(mPath+mFilename, cv::FileStorage::WRITE);
	}

	intptr_t addObject(const ISerializable *obj);
	std::string addImage(const std::string &prefix, const cv::Mat &img);
	void serializeAll();

protected:
	std::string mPath;
	std::string mFilename;
	cv::FileStorage mFS;

	std::stack<const ISerializable *> mObjectsToSerialize;
	std::unordered_set<const ISerializable *> mObjectsAdded;

	std::unordered_map<std::string, int> mImageCounter; //<prefix, nextIdx>

	void serializeObject(const ISerializable &obj);
};

class Deserializer
{
public:
	//Path can be empty, else it must include the path separator. Absolute filename path=path+filename.
	//Images will be stored in the same path but with filenames according to the prefix.
	void open(const std::string &path, const std::string &filename)
	{
		mPath = path;
		mFilename = filename;
		mFS.open(mPath + mFilename, cv::FileStorage::READ);
	}

	void deserialize();

	//getObject & getObjectForOwner
	//These two get an object based on its old address
	template<class T>
	T *getObject(const cv::FileNode &addrNode);

	template<class T>
	std::unique_ptr<T> getObjectForOwner(const cv::FileNode &addrNode);

	//getObject & getObjectForOwner
	//These two get the first object of a given type
	template<class T>
	T *getObject();

	template<class T>
	std::unique_ptr<T> getObjectForOwner();


	cv::Mat getImage(const std::string &name);

	template<class T>
	std::unique_ptr<T> releaseOwnership(T *object);

protected:
	struct ObjectInfo
	{
		ObjectInfo(ISerializable *object_, const cv::FileNode &node_) :
		object(object_), node(node_), isDeserialized(false)
		{
		}

		ISerializable *object;
		const cv::FileNode node;
		bool isDeserialized;

		void ensureDeserilized(Deserializer &s);
	};

	std::string mPath;
	std::string mFilename;
	cv::FileStorage mFS;
	
	std::unordered_map<ISerializable *, std::unique_ptr<ISerializable>> mObjectsOwned;
	std::unordered_map<intptr_t, ObjectInfo> mAddressMap;

	ISerializable *createObject(const cv::FileNode &node);
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Additional write functions
template<int M, int N> static inline cv::FileStorage &operator <<(cv::FileStorage &fs,
                                                                  const cv::Matx<float, M, N> &value)
{
    fs << "[";
    fs.writeRaw("f", (uchar *)value.val, sizeof(value.val));
    return fs << "]";
}

template<int M, int N> static inline void operator >>(const cv::FileNode &n, cv::Matx<float, M, N> &value)
{
    n.readRaw("f", (uchar *)value.val, sizeof(value.val));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template<class T>
T *Deserializer::getObject(const cv::FileNode &addrNode)
{
	intptr_t addr;
	addrNode >> addr;
	if (addr == 0)
		return nullptr;

	auto it = mAddressMap.find(addr);
	if (it == mAddressMap.end())
	{
		throw std::runtime_error("Requested object is not in the address map.");
	}

	it->second.ensureDeserilized(*this);

	return dynamic_cast<T*>(it->second.object);
}

template<class T>
std::unique_ptr<T> Deserializer::getObjectForOwner(const cv::FileNode &addrNode)
{
	T *obj = getObject<T>(addrNode);
	return releaseOwnership(obj);
}

template<class T>
T *Deserializer::getObject()
{
	for (auto &obj : mAddressMap)
	{
		T *p = dynamic_cast<T*>(obj.second.object);
		if (p)
		{
			obj.second.ensureDeserilized(*this);
			return p;
		}
	}
	return nullptr;
}

template<class T>
std::unique_ptr<T> Deserializer::getObjectForOwner()
{
	T *obj = getObject<T>();
	return releaseOwnership(obj);
}

template<class T>
std::unique_ptr<T> Deserializer::releaseOwnership(T *object)
{
	std::unique_ptr<T> res;
	
	if (object == nullptr)
		return res;

	auto it = mObjectsOwned.find(object);
	if (it == mObjectsOwned.end())
	{
		throw std::runtime_error("Object not owned, cannot release.");
	}

	res.reset(dynamic_cast<T*>(it->second.release()));
	mObjectsOwned.erase(it);

	return res;
}

} /* namespace dtslam */

#endif /* SERIALIZER_H_ */
