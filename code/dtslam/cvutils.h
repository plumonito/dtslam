/*
 * cvutils.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef CVUTILS_H_
#define CVUTILS_H_

#include <opencv2/core.hpp>
#include <cassert>

namespace dtslam
{

class cvutils
{
public:
	static cv::Point3f PointToHomogenous(const cv::Point2f &p)
	{
		return cv::Point3f(p.x,p.y,1);
	}
	static cv::Point3f PointToHomogenous(const cv::Point2f &p, float p3)
	{
		return cv::Point3f(p3*p.x,p3*p.y,p3);
	}
	static cv::Point3f PointToHomogenousUnitNorm(const cv::Point2f &p)
	{
		float factorInv = 1.0f/std::sqrt(p.x*p.x+p.y*p.y+1);
		return cv::Point3f(p.x*factorInv, p.y*factorInv, factorInv);
	}

	static cv::Vec4f PointToHomogenous(const cv::Point3f &p)
	{
		return cv::Vec4f(p.x,p.y,p.z,1);
	}

	static cv::Point2f NormalizePoint(const cv::Point3f &p)
	{
		return cv::Point2f(p.x/p.z, p.y/p.z);
	}
	static cv::Point3f NormalizePoint(const cv::Vec4f &p)
	{
		return cv::Point3f(p[0]/p[3], p[1]/p[3], p[2]/p[3]);
	}

	static cv::Point3f PointToUnitNorm(const cv::Point3f &p)
	{
		float factorInv = 1.0f/std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
		return cv::Point3f(p.x*factorInv, p.y*factorInv, p.z*factorInv);
	}


	static float NormSq(const cv::Point2f &p)
	{
		return p.x*p.x+p.y*p.y;
	}
	static float NormSq(const cv::Point3f &p)
	{
		return p.x*p.x+p.y*p.y+p.z*p.z;
	}
	template<class T,int N>
	static T NormSq(const cv::Vec<T,N> &p)
	{
		T sum=0;
		for(int i=0; i<N; ++i)
		{
			sum += p[i]*p[i];
		}
		return sum;
	}

	template<class T,int N>
	static cv::Vec<T,N> VecToUnitNorm(const cv::Vec<T,N> &p)
	{
		const T factorInv = T(1.0)/std::sqrt(NormSq(p));
		cv::Vec<T,N> res;
		for(int i=0; i<N; ++i)
			res[i] = p[i]*factorInv;
		return res;
	}

	static float PointDistSq(const cv::Point2f &p1, const cv::Point2f &p2)
	{
		const float dx=p1.x-p2.x;
		const float dy=p1.y-p2.y;
		return dx*dx+dy*dy;
	}
	static float PointDistSq(const cv::Point3f &p1, const cv::Point3f &p2)
	{
		const float dx=p1.x-p2.x;
		const float dy=p1.y-p2.y;
		const float dz=p1.z-p2.z;
		return dx*dx+dy*dy+dz*dz;
	}

	template<class T, int N>
	static float PointDistSq(const cv::Vec<T,N> &p1, const cv::Vec<T,N> &p2)
	{
		float sum=0;
		for(int i=0; i<N; ++i)
		{
			const float di = p1[i]-p2[i];
			sum += di*di;
		}
		return sum;
	}

	template<class T, int cn>
	static cv::Vec<T,cn> matrixPointHomogeneous(const cv::Matx<T,cn,cn> &m, const cv::Vec<T,cn-1> &v)
	{
		cv::Vec<T,cn> res;
		for(int j=0; j<cn; ++j)
		{
			res[j] = m(j,cn-1);
			for(int i=0; i<cn-1; ++i)
				res[j]+=m(j,i)*v[i];
		}
		return res;
	}

	template<class T>
	static cv::Vec<T,3> matrixPointHomogeneous(const cv::Matx<T,3,3> &m, const cv::Point_<T> &v)
	{
		return cv::Vec<T,3>(m(0,0)*v.x+m(0,1)*v.y+m(0,2),
							m(1,0)*v.x+m(1,1)*v.y+m(1,2),
							m(2,0)*v.x+m(2,1)*v.y+m(2,2));
	}

	static cv::Vec3f NormalizeLine(const cv::Vec3f &line)
	{
		const float factorInv = 1.0f/sqrtf(line[0]*line[0] + line[1]*line[1]);
		return line*factorInv;
	}

	static cv::Point2f HomographyPoint(const cv::Matx33f &h, const cv::Point2f &p)
	{
		const float p2 = (h(2,0)*p.x + h(2,1)*p.y + h(2,2));
		const float p0 = h(0,0)*p.x + h(0,1)*p.y + h(0,2);
		const float p1 = h(1,0)*p.x + h(1,1)*p.y + h(1,2);
		return cv::Point2f(p0/p2, p1/p2);
	}

	static cv::Point2f AffinePoint(const cv::Matx23f &h, const cv::Point2f &p)
	{
		const float p0 = h(0,0)*p.x + h(0,1)*p.y + h(0,2);
		const float p1 = h(1,0)*p.x + h(1,1)*p.y + h(1,2);
		return cv::Point2f(p0, p1);
	}
	static cv::Point3f AffinePoint(const cv::Matx23f &h, const cv::Point3f &p)
	{
		const float p0 = h(0,0)*p.x + h(0,1)*p.y + p.z*h(0,2);
		const float p1 = h(1,0)*p.x + h(1,1)*p.y + p.z*h(1,2);
		return cv::Point3f(p0, p1, p.z);
	}
	static cv::Point2f AffinePoint(const cv::Matx33f &h, const cv::Point2f &p)
	{
		assert(h(2,0)==0 && h(2,1)==0 && h(2,2)==1);
		const float p0 = h(0,0)*p.x + h(0,1)*p.y + h(0,2);
		const float p1 = h(1,0)*p.x + h(1,1)*p.y + h(1,2);
		return cv::Point2f(p0, p1);
	}

	static cv::Vec3f LineAffine(const cv::Vec3f &line, const cv::Matx23f &h)
	{
		const float l0 = line[0]*h(0,0) + line[1]*h(1,0);
		const float l1 = line[0]*h(0,1) + line[1]*h(1,1);
		const float l2 = line[0]*h(0,2) + line[1]*h(1,2) + line[2];
		return cv::Vec3f(l0, l1, l2);
	}

	static cv::Matx23f AffineAffine(const cv::Matx23f &aff1, const cv::Matx23f &aff2)
	{
		const float a00 = aff1(0,0)*aff2(0,0) + aff1(0,1)*aff2(1,0);
		const float a01 = aff1(0,0)*aff2(0,1) + aff1(0,1)*aff2(1,1);
		const float a10 = aff1(1,0)*aff2(0,0) + aff1(1,1)*aff2(1,0);
		const float a11 = aff1(1,0)*aff2(0,1) + aff1(1,1)*aff2(1,1);
		const float a02 = aff1(0,0)*aff2(0,2) + aff1(0,1)*aff2(1,2) + aff1(0,2);
		const float a12 = aff1(1,0)*aff2(0,2) + aff1(1,1)*aff2(1,2) + aff1(1,2);

		return cv::Matx23f(a00,a01,a02,a10,a11,a12);
	}

	static cv::Matx34f CatRT(const cv::Matx33f &R, const cv::Vec3f &t)
	{
		return cv::Matx34f(R(0,0),R(0,1),R(0,2),t[0],
				R(1,0),R(1,1),R(1,2),t[1],
				R(2,0),R(2,1),R(2,2),t[2]);
	}

	static void CalculateDerivatives(const cv::Mat1b &img, cv::Mat1s &dx, cv::Mat1s &dy);

	static cv::Matx33f SkewSymmetric(const cv::Vec3f &t)
	{
		return cv::Matx33f(0,-t[2],t[1],  t[2],0,-t[0],  -t[1],t[0],0);
	}

	template <class T, class TBool>
	static void SplitVector(const std::vector<T> &src, const std::vector<TBool> &isLeft, std::vector<T> &left)
	{
		assert(src.size() == isLeft.size());
		for(int i=0; i<(int)src.size(); ++i)
		{
			if(isLeft[i])
			{
				left.push_back(src[i]);
			}
		}
	}

	template <class T, class TBool>
	static void SplitVector(const std::vector<T> &src, const std::vector<TBool> &isLeft, std::vector<T> &left, std::vector<T> &right)
	{
		assert(src.size() == isLeft.size());
		for(int i=0; i<(int)src.size(); ++i)
		{
			if(isLeft[i])
			{
				left.push_back(src[i]);
			}
			else
			{
				right.push_back(src[i]);
			}
		}
	}

	static cv::Matx33f RotationX(float angle)
	{
		return cv::Matx33f(cos(angle), 0, sin(angle), 0, 1, 0, -sin(angle), 0, cos(angle));
	}

	static cv::Matx33f RotationY(float angle)
	{
		return cv::Matx33f(1, 0, 0, 0, cos(angle), -sin(angle), 0, sin(angle), cos(angle));
	}

    static void DownsampleImage(const cv::Mat &img, cv::Mat &res, int count);

    //A point in the line is given by X=linePoint+alpha*lineDir
    //A point in the plane is given by (X-planePoint)*planeNormal=0
    static cv::Vec3f linePlaneIntersection(const cv::Vec3f &linePoint, const cv::Vec3f &lineDir, const cv::Vec3f &planePoint, const cv::Vec3f &planeNormal)
    {
		const float alpha = planeNormal.dot(planePoint-linePoint) / planeNormal.dot(lineDir);
    	return linePoint + alpha*lineDir;
    }

	//Equations: dot( (x - planePoint), planeNormal) = 0
	//x = linePoint + lineDirection*alpha
	static cv::Point3f intersectPlaneLine(const cv::Vec3f &planeNormal, const cv::Point3f &planePoint, const cv::Vec3f &lineDirection, const cv::Point3f &linePoint)
	{
		float alpha = planeNormal.dot(planePoint-linePoint) / planeNormal.dot(lineDirection);
		return linePoint + cv::Point3f(alpha*lineDirection);
	}
};

}

#endif /* CVUTILS_H_ */
