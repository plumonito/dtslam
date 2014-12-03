/*
 * Pose3D.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef POSE3D_H_
#define POSE3D_H_

#include "cvutils.h"
#include <memory>
#include "Serializer.h"

namespace dtslam {

class Pose3D: public ISerializable
{
public:
	virtual ~Pose3D() {}

	virtual std::unique_ptr<Pose3D> copy() const = 0;

	virtual cv::Matx33f getRotation() const = 0;
	virtual cv::Vec3f getTranslation() const = 0;

	virtual int getDoF() const = 0;
	virtual void setFromArray(const std::vector<double> &array)=0;
	virtual void copyToArray(std::vector<double> &array) const=0;

	cv::Matx34f getRt() const {return cvutils::CatRT(getRotation(), getTranslation());}
	cv::Vec3f getCenter() const {return -getRotation().t()*getTranslation();}

	virtual cv::Point3f apply(const cv::Point3f &p) const=0;
	virtual cv::Point3f applyInv(const cv::Point3f &p) const=0;
	virtual cv::Vec4f apply(const cv::Vec4f &p) const=0;
	virtual cv::Vec4f applyInv(const cv::Vec4f &p) const=0;

	/////////////////////////////////////////////////////
	// Because templates cannot be virtual, these static methods dispatch to the appropriate derived class
	//Apply to ceres variables (point is in 3D)
	template<class T>
	static void Apply3D(const Pose3D *pose, const T * const point, T*result);

	//Apply to ceres variables (point is in homogeneous 4D)
	template<class T>
	static void Apply4D(const Pose3D *pose, const T * const point, T*result);

	//Apply inverse to ceres variables (point is in 3D)
	template<class T>
	static void ApplyInv3D(const Pose3D *pose, const T * const point, T*result);

	//Apply inverse to ceres variables (point is in homogeneous 4D)
	template<class T>
	static void ApplyInv4D(const Pose3D *pose, const T * const point, T*result);

protected:
};

class FullPose3D: public Pose3D
{
public:
	FullPose3D():
		mRotation(cv::Matx33f::eye()), mTranslation(0,0,0)
	{
	}

	FullPose3D(const cv::Matx33f &R, const cv::Vec3f &T):
		mRotation(R), mTranslation(T)
	{
	}

	FullPose3D(const Pose3D &pose):
		FullPose3D(pose.getRotation(), pose.getTranslation())
	{
	}

	std::unique_ptr<Pose3D> copy() const {return std::unique_ptr<Pose3D>(new FullPose3D(*this));}

	cv::Matx33f &getRotationRef() {return mRotation;}
	const cv::Matx33f &getRotationRef() const {return mRotation;}
	cv::Vec3f &getTranslationRef() {return mTranslation;}
	const cv::Vec3f &getTranslationRef() const {return mTranslation;}

	void set(const cv::Matx33f &R, const cv::Vec3f &t) {mRotation = R; mTranslation = t;}
	void set(const Pose3D &pose) {set(pose.getRotation(), pose.getTranslation());}
	void setRotation(const cv::Matx33f &R) {mRotation = R;}
	void setTranslation(const cv::Vec3f &t) {mTranslation = t;}

	int getDoF() const {return 6;}
	void setFromArray(const std::vector<double> &array);
	void copyToArray(std::vector<double> &array) const;

	cv::Point3f apply(const cv::Point3f &p) const
	{
		return mRotation*p + cv::Point3f(mTranslation);
	}

	cv::Vec4f apply(const cv::Vec4f &p) const
	{
		return cv::Vec4f(mRotation(0,0)*p[0]+mRotation(0,1)*p[1]+mRotation(0,2)*p[2] + mTranslation[0]*p[3],
				mRotation(1,0)*p[0]+mRotation(1,1)*p[1]+mRotation(1,2)*p[2] + mTranslation[1]*p[3],
				mRotation(2,0)*p[0]+mRotation(2,1)*p[1]+mRotation(2,2)*p[2] + mTranslation[2]*p[3],
				p[3]);
	}

	cv::Point3f applyInv(const cv::Point3f &p) const
	{
		return mRotation.t()*(p - cv::Point3f(mTranslation));
	}

	cv::Vec4f applyInv(const cv::Vec4f &p) const
	{
		cv::Vec3f pt(p[0]-mTranslation[0]*p[3],
				p[1]-mTranslation[1]*p[3],
				p[2]-mTranslation[2]*p[3]);
		return cv::Vec4f(mRotation(0,0)*pt[0]+mRotation(1,0)*pt[1]+mRotation(2,0)*pt[2],
				mRotation(0,1)*pt[0]+mRotation(1,1)*pt[1]+mRotation(2,1)*pt[2],
				mRotation(0,2)*pt[0]+mRotation(1,2)*pt[1]+mRotation(2,2)*pt[2],
				p[3]);
	}

	//Apply to ceres variables (point is in 3D)
	template<class T>
	static void Apply3D(const T * const poseR, const T * const poseT, const T * const point, T*result);

	//Apply to ceres variables (point is in 3D)
	template<class T>
	void apply3D(const T * const point, T*result) const;

	//Apply to ceres variables (point is in homogeneous 4D)
	template<class T>
	void apply4D(const T * const point, T*result) const;

	//Apply inverse to ceres variables (point is in 3D)
	template<class T>
	void applyInv3D(const T * const point, T*result) const;

	//Apply inverse to ceres variables (point is in homogeneous 4D)
	template<class T>
	void applyInv4D(const T * const point, T*result) const;


	//Creates a FullPose3D that transforms directly from ref coordinates to img coordinates
	static FullPose3D MakeRelativePose(const Pose3D &ref, const Pose3D &img)
	{
		const cv::Matx33f refR = ref.getRotation();
		const cv::Vec3f refT = ref.getTranslation();
		const cv::Matx33f imgR = img.getRotation();
		const cv::Vec3f imgT = img.getTranslation();
		const cv::Matx33f refRt = refR.t();

		const cv::Matx33f Rab = imgR * refRt;
		const cv::Vec3f tab = imgT - Rab*refT;
		return FullPose3D(Rab, tab);
	}

	//Creates a FullPose3D that transforms directly from ref coordinates to img coordinates
	//Templated version for ceres
	template<class T>
	static void MakeRelativePose(const T * const refR, const T * const refT, const T * const imgR, const T * const imgT, T *relR, T *relT);

	/////////////////////////////////////
	// Serialization
	static const std::string GetTypeName() { return "FullPose3D"; }
	const std::string getTypeName() const {return GetTypeName();}
	void serialize(Serializer &s, cv::FileStorage &fs) const;
	void deserialize(Deserializer &s, const cv::FileNode &node);

protected:
	cv::Matx33f mRotation;
	cv::Vec3f mTranslation;

	cv::Matx33f getRotation() const {return mRotation;}
	cv::Vec3f getTranslation() const {return mTranslation;}
};

class RelativeRotationPose3D: public Pose3D
{
public:
	RelativeRotationPose3D(): mBasePose(NULL)
	{
	}

	RelativeRotationPose3D(const Pose3D * const basePose, const cv::Matx33f &absoluteR)
	{
		setFromAbsolute(basePose, absoluteR);
	}

	std::unique_ptr<Pose3D> copy() const {return std::unique_ptr<Pose3D>(new RelativeRotationPose3D(*this));}

	cv::Matx33f getRotation() const { return mRelativeRotation*mBasePose->getRotation(); }
	cv::Vec3f getTranslation() const {return mRelativeRotation*mBasePose->getTranslation();}

	void setFromAbsolute(const Pose3D * const basePose, const cv::Matx33f &R)
	{
		mBasePose = NULL;

		const RelativeRotationPose3D *relrptr = dynamic_cast<const RelativeRotationPose3D*>(basePose);
		if (relrptr)
			mBasePose = relrptr->getBasePose();
		
		const FullPose3D *fullptr = dynamic_cast<const FullPose3D*>(basePose);
		if (fullptr)
			mBasePose = fullptr;

		assert(mBasePose);
		mRelativeRotation = R * mBasePose->getRotation().t();
	}

	const Pose3D *getBasePose() const {return mBasePose;}
	const cv::Matx33f &getRelativeRotation() const {return mRelativeRotation;}

	int getDoF() const {return 3;}
	void setFromArray(const std::vector<double> &array);
	void copyToArray(std::vector<double> &array) const;

	cv::Point3f apply(const cv::Point3f &p) const
	{
		return mRelativeRotation*mBasePose->apply(p);
	}

	cv::Vec4f apply(const cv::Vec4f &p) const
	{
		cv::Vec4f pb = mBasePose->apply(p);
		return cv::Vec4f(mRelativeRotation(0,0)*pb[0]+mRelativeRotation(0,1)*pb[1]+mRelativeRotation(0,2)*pb[2],
				mRelativeRotation(1,0)*pb[0]+mRelativeRotation(1,1)*pb[1]+mRelativeRotation(1,2)*pb[2],
				mRelativeRotation(2,0)*pb[0]+mRelativeRotation(2,1)*pb[1]+mRelativeRotation(2,2)*pb[2],
				pb[3]);
	}

	cv::Point3f applyInv(const cv::Point3f &p) const
	{
		return mBasePose->applyInv(mRelativeRotation.t()*p);
	}

	cv::Vec4f applyInv(const cv::Vec4f &p) const
	{
		cv::Vec4f pb(mRelativeRotation(0,0)*p[0]+mRelativeRotation(1,0)*p[1]+mRelativeRotation(2,0)*p[2],
				mRelativeRotation(0,1)*p[0]+mRelativeRotation(1,1)*p[1]+mRelativeRotation(2,1)*p[2],
				mRelativeRotation(0,2)*p[0]+mRelativeRotation(1,2)*p[1]+mRelativeRotation(2,2)*p[2],
				p[3]);
		return mBasePose->applyInv(pb);
	}

	//Apply to ceres variables (point is in 3D)
	template<class T>
	void apply3D(const T * const point, T*result) const;

	//Apply to ceres variables (point is in homogeneous 4D)
	template<class T>
	void apply4D(const T * const point, T*result) const;

	//Apply inverse to ceres variables (point is in 3D)
	template<class T>
	void applyInv3D(const T * const point, T*result) const;

	//Apply inverse to ceres variables (point is in homogeneous 4D)
	template<class T>
	void applyInv4D(const T * const point, T*result) const;

	/////////////////////////////////////
	// Serialization
	static const std::string GetTypeName() { return "RelativeRotationPose3D"; }
	const std::string getTypeName() const { return GetTypeName(); }
	void serialize(Serializer &s, cv::FileStorage &fs) const;
	void deserialize(Deserializer &s, const cv::FileNode &node);

protected:
	const Pose3D * mBasePose;
	cv::Matx33f mRelativeRotation;
};

class RelativePose3D: public Pose3D
{
public:
	RelativePose3D():
		mBasePose(NULL)
	{
	}
	RelativePose3D(const Pose3D * const basePose):
		mBasePose(basePose), mRelativePose(cv::Matx33f::eye(), cv::Vec3f(0,0,0))
	{
	}

	std::unique_ptr<Pose3D> copy() const {return std::unique_ptr<Pose3D>(new RelativePose3D(*this));}

	cv::Matx33f getRotation() const {return mBasePose->getRotation()*mRelativePose.getRotationRef();}
	cv::Vec3f getTranslation() const {return mBasePose->getRotation()*mRelativePose.getTranslationRef() + mBasePose->getTranslation();}

	const Pose3D &getBasePose() const {return *mBasePose;}
	void setBasePose(const Pose3D *base)
	{
		if(mBasePose)
		{
			setFromAbsolute(base, getRotation(), getTranslation());
		}
		else
		{
			mBasePose = base;
		}
	}

	const FullPose3D &getRelativePose() const {return mRelativePose;}

	//Sets this to be the same as the absolute pose given, but stores it internally as a relative pose
	//After this getRotation()==rotation and getTranslation()==translation with the current base pose
	void setFromAbsolute(const Pose3D *base, const cv::Matx33f &rotation, const cv::Vec3f &translation)
	{
		mBasePose = base;
		setFromAbsolute(rotation, translation);
	}
	void setFromAbsolute(const cv::Matx33f &rotation, const cv::Vec3f &translation);
	void setFromAbsolute(const Pose3D *base, const Pose3D &pose) {setFromAbsolute(base, pose.getRotation(), pose.getTranslation());}
	void setFromAbsolute(const Pose3D &pose) {setFromAbsolute(pose.getRotation(), pose.getTranslation());}

	int getDoF() const {return 6;}
	void setFromArray(const std::vector<double> &array);
	void copyToArray(std::vector<double> &array) const;

	cv::Point3f apply(const cv::Point3f &p) const
	{
		return mRelativePose.apply(mBasePose->apply(p));
	}

	cv::Vec4f apply(const cv::Vec4f &p) const
	{
		return mRelativePose.apply(mBasePose->apply(p));
	}

	cv::Point3f applyInv(const cv::Point3f &p) const
	{
		return mBasePose->applyInv(mRelativePose.applyInv(p));
	}

	cv::Vec4f applyInv(const cv::Vec4f &p) const
	{
		return mBasePose->applyInv(mRelativePose.applyInv(p));
	}

	//Apply to ceres variables (point is in 3D)
	template<class T>
	void apply3D(const T * const point, T*result) const;

	//Apply to ceres variables (point is in homogeneous 4D)
	template<class T>
	void apply4D(const T * const point, T*result) const;

	//Apply inverse to ceres variables (point is in 3D)
	template<class T>
	void applyInv3D(const T * const point, T*result) const;

	//Apply inverse to ceres variables (point is in homogeneous 4D)
	template<class T>
	void applyInv4D(const T * const point, T*result) const;

	/////////////////////////////////////
	// Serialization
	static const std::string GetTypeName() { return "RelativePose3D"; }
	const std::string getTypeName() const { return GetTypeName(); }
	void serialize(Serializer &s, cv::FileStorage &fs) const;
	void deserialize(Deserializer &s, const cv::FileNode &node);

protected:
	const Pose3D * mBasePose;
	FullPose3D mRelativePose;
};

} /* namespace dtslam */

#endif /* POSE3D_H_ */
