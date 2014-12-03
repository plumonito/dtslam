/*
 * Serializer.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include <iostream>
#include <iomanip>
#include <opencv2/highgui.hpp>

#include "Serializer.h"
#include "Pose3D.h"
#include "SlamMap.h"
#include "SlamKeyFrame.h"


namespace dtslam {

intptr_t Serializer::addObject(const ISerializable *obj)
{
	if (obj == nullptr)
		return 0;

	auto it=mObjectsAdded.insert(obj);
	if(it.second)
	{
		mObjectsToSerialize.push(obj);
	}
	return (intptr_t)obj;
}

std::string Serializer::addImage(const std::string &prefix, const cv::Mat &img)
{
	//Get counter
	auto it = mImageCounter.insert(std::make_pair(prefix, -1));
	int &index = it.first->second;
	index++;

	//Filename
	std::stringstream ss;
	ss << prefix << std::setw(8) << std::setfill('0') << index << ".jpg";
	
	cv::imwrite(mPath + ss.str(), img);

	return ss.str();
}

void Serializer::serializeAll()
{
	while(!mObjectsToSerialize.empty())
	{
		const ISerializable *obj = mObjectsToSerialize.top();
		mObjectsToSerialize.pop();

		serializeObject(*obj);
	}
}

void Serializer::serializeObject(const ISerializable &obj)
{
	intptr_t addr = (intptr_t)&obj;

	//Create a name for the object
	std::stringstream ss;
	ss << obj.getTypeName() << addr;

	mFS << ss.str();
	mFS << "{";
	mFS << "type" << obj.getTypeName();
	mFS << "addr" << addr;
	obj.serialize(*this, mFS);
	mFS << "}";
}


void Deserializer::ObjectInfo::ensureDeserilized(Deserializer &s)
{
	if (!isDeserialized)
	{
		isDeserialized = true;
		object->deserialize(s, node);
	}
}

void Deserializer::deserialize()
{
	cv::FileNode root = mFS.root();

	//Create all objects
	for(auto it=root.begin(),end=root.end(); it!=end; ++it)
	{
		const cv::FileNode &node=*it;
		createObject(node);
	}

	//Deserialize all
	for(auto &obj : mAddressMap)
	{
		obj.second.ensureDeserilized(*this);
	}
}

ISerializable *Deserializer::createObject(const cv::FileNode &node)
{
	std::string type;
	intptr_t addr;

	node["type"] >> type;
	node["addr"] >> addr;

	std::unique_ptr<ISerializable> obj;
	ISerializable *objP;

	//Create
	if(type == FullPose3D::GetTypeName())
		obj.reset(new FullPose3D());
	else if(type == RelativeRotationPose3D::GetTypeName())
		obj.reset(new RelativeRotationPose3D());
	else if(type == RelativePose3D::GetTypeName())
		obj.reset(new RelativePose3D());
	else if (type == SlamKeyFrame::GetTypeName())
		obj.reset(new SlamKeyFrame());
	else if (type == SlamMap::GetTypeName())
		obj.reset(new SlamMap());
	else if (type == SlamRegion::GetTypeName())
		obj.reset(new SlamRegion());
	else if (type == SlamFeature::GetTypeName())
		obj.reset(new SlamFeature());
	else if (type == SlamFeatureMeasurement::GetTypeName())
		obj.reset(new SlamFeatureMeasurement());
	else if (type == CameraModel::GetTypeName())
		obj.reset(new CameraModel());
	else
		throw std::runtime_error("Unknown type");
	objP = obj.get();

	//Remember
	mAddressMap.insert(std::make_pair(addr, Deserializer::ObjectInfo(objP,node)));
	mObjectsOwned.insert(std::pair<ISerializable*, std::unique_ptr<ISerializable>>(objP, std::move(obj)));

	return objP;
}

cv::Mat Deserializer::getImage(const std::string &name)
{
	return cv::imread(mPath + name);
}

} /* namespace dtslam */
