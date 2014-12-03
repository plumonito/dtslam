#include <limits.h>
#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>

#include "nvslam/CameraModel.h"

using namespace nvslam;

TEST(RadialCameraDistortionTest, undistortOne)
{
	RadialCameraDistortionModel dist;
	//dist.init(0.16262,-0.67445);
	dist.init( 0.1204,   -0.2117);

	cv::Point3f xn(-0.4952,-0.3689,1);
	cv::Point2f xd = dist.distortPoint(xn);
	cv::Point3f xnn = dist.undistortPoint(xd);
	xnn *= 1/xnn.z;
	ASSERT_FLOAT_EQ(xnn.x, xn.x);
	ASSERT_FLOAT_EQ(xnn.y, xn.y);

	const int maxi=100000;
	for(int i=0; i<maxi; i++){
	cv::Point2f xd(0.1f*i / maxi,0);
	cv::Point3f xn = dist.undistortPoint(xd);
	cv::Point2f xdd = dist.distortPoint(xn);
	ASSERT_FLOAT_EQ(xdd.x, xd.x);
	ASSERT_FLOAT_EQ(xdd.y, xd.y);
	}
}

bool IsPointEquali(const cv::Point2i &a, const cv::Point2i &b)
{
	return a.x==b.x && a.y==b.y;
}

bool IsPointEqualf(const cv::Point2f &a, const cv::Point2f &b)
{
	return fabs(a.x-b.x)<0.01f && fabs(a.y-b.y)<0.01f;
}

//Note: if this test fails try increasing the iteration count in RadialCameraDistortionModel::undistortPoint()
TEST(RadialCameraDistortionTest, undistortAll)
{
	cv::Size2i imageSize(1920, 1080);
	CameraModel_<RadialCameraDistortionModel> camera;
	camera.init(1759.95830,   1758.22390, 963.80029,   541.04811, 1920, 1080);
	camera.getDistortionModel().init(0.16262,-0.67445);

	//camera.useUndistortionLUT(imageSize);

	for(int v=0; v<imageSize.height; v++)
	{
		for(int u=0; u<imageSize.width; u++)
		{
			cv::Point2i uv(u,v);
			cv::Point3f xn = camera.unprojectToWorld(uv);
			cv::Point2f uvp = camera.projectFromWorld(xn);
			EXPECT_PRED2(IsPointEqualf, uv, uvp);
		}
	}
}
