/*
 * ARWindow.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef ARWINDOW_H_
#define ARWINDOW_H_

#include "dtslam/Pose3D.h"
#include "dtslam/CameraModel.h"
#include "BaseWindow.h"

namespace dtslam
{

class SlamMap;
class SlamRegion;
class Slam2DSection;
class FrameTrackingData;
struct PoseEstimatorData;
class PoseTracker;

class ARWindow: public BaseWindow
{
public:
	ARWindow():
		BaseWindow("ARWindow"), mDisplayType(EDisplayType::ShowMatches)
	{}

	bool init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize);
	void showHelp() const;

	void updateState();
    void resize();

//    void touchDown(int id, int x, int y);
//    void touchMove(int x, int y);
//    void touchUp(int id, int x, int y);

    void draw();

protected:
    ViewportTiler mTiler;

    SlamMap *mMap;
    PoseTracker *mTracker;

	enum class EDisplayType
	{
		ShowMatches,
		ShowStableFeatures,
		ShowAllFeatures
	};
	EDisplayType mDisplayType;

    const CameraModel *mTrackerCamera;
	const Pose3D *mTrackerPose;

	//Draw data
	std::vector<cv::Point2f> mImagePoints;
	std::vector<cv::Vec4f> mImagePointColors;

	std::vector<cv::Vec4f> mFeatureVertices;
	std::vector<cv::Vec4f> mFeatureColors;

	std::vector<unsigned int> mCubeTriangleIndices;
	std::vector<cv::Vec4f> mCubeVertices;
	std::vector<cv::Vec4f> mCubeColors;
	std::vector<cv::Vec3f> mCubeNormals;

	void toggleDisplayType();
};

} /* namespace dtslam */

#endif /* ARWINDOW_H_ */
