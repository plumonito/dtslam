/*
 * MapWindow.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef MAPWINDOW_H_
#define MAPWINDOW_H_

#include <future>
#include "dtslam/Pose3D.h"
#include "dtslam/CameraModel.h"
#include "BaseWindow.h"
#include "../shaders/StaticColors.h"

namespace dtslam {

class SlamMap;
class SlamRegion;
class Slam2DSection;
class PoseTracker;

class MapWindow: public BaseWindow
{
public:
	MapWindow():
		BaseWindow("MapWindow"),
		mDrawFeaturePatches(false),
		mPointSize(4),
		mActiveDragType(EDragType::NoDragging),
		mRegionColors({StaticColors::Blue(), StaticColors::Purple(), StaticColors::Green(), StaticColors::Yellow()})
	{
	}

	bool init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize);
	void showHelp() const;

	void updateState();
    void resize();

    void touchDown(int id, int x, int y);
    void touchMove(int x, int y);
    void touchUp(int id, int x, int y);

    void draw();

protected:
    SlamMap *mMap;
    SlamRegion *mRegion;
    PoseTracker *mTracker;

    bool mDrawFeaturePatches;
    float mPointSize;

    CameraModel_<NullCameraDistortionModel> mViewerCamera;
	FullPose3D mViewerPose;

	std::vector<const SlamFeature *> mFeaturesInView;
	std::unordered_map<const SlamKeyFrame *, std::unique_ptr<TextureHelper>> mFrameTextures;

	//Dragging
	enum class EDragType
	{
		NoDragging=0,
		DraggingRotation,
		DraggignTranslation,
		DraggingCube
	};
	static const int kTranslateScale=20;
	EDragType mActiveDragType;
	FullPose3D mDragStartingPose;
    cv::Point2f mDragOrigin; //The origin of the dragging in pixel units
    cv::Point2f mDragEnd; //The end of the dragging in pixel units (updates as the mouse moves even before the button is released)

    void togglePatches() {mDrawFeaturePatches=!mDrawFeaturePatches;}
    void increasePointSize() {mPointSize++;}
    void decreasePointSize() {mPointSize--; if(mPointSize<0) mPointSize=0;}

	void zoomIn();
	void zoomOut();
	void zoom(float ammount);

    void ensureValidRegion();
    void selectNextRegion();

    void performBA();
    std::future<void> mPerformBAFuture;

    static void PerformBATask(MapWindow *window);
    void performBATask();

    void forceNewKeyFrame();

    void startCube();
    void updateCube(const cv::Point2f &origin, const cv::Point2f &end);

    const TextureHelper &getFrameTexture(const SlamKeyFrame &frame);

    bool isFeatureMatched(const SlamFeature &feature);

	//Drawing stuff
	float mMapDrawScale;
	const std::vector<cv::Vec4f> mRegionColors;

    //Drawing of feature
    struct DrawFeatureData
    {
    	cv::Vec4f vertices[4];
    	cv::Vec4f center;

	    cv::Vec4f solidColor;

	    bool useTex;
	    unsigned int texTarget;
	    unsigned int texId;
	    cv::Vec2f texCoordinates[4];
    };
    std::vector<DrawFeatureData> mFeaturesToDraw;
    std::vector<DrawFeatureData> mMatchedFeaturesToDraw; //These are in a separate vector so they are drawn after and are always visible.
    void drawFeature(const DrawFeatureData &data);

	//Drawing of a camera frustum on the map
	struct DrawFrustumData
	{
		cv::Point3f tl;
	    cv::Point3f tr;
	    cv::Point3f bl;
	    cv::Point3f br;

	    cv::Point3f center;

	    cv::Vec4f color;

	    bool useTex;
	    unsigned int texTarget;
	    unsigned int texId;
	};
	std::vector<DrawFrustumData> mFrustumsToDraw;

	std::vector<cv::Vec4f> mPoseLog;

	DrawFrustumData prepareFrameFrustum(const Pose3D &pose, const CameraModel &camera, int regionId, bool useTex, unsigned int texTarget=0, unsigned int texID=0);
    void drawFrameFrustum(const DrawFrustumData &data);

    /**
     * @brief Draw function for the view frustum of each keyframe
     */
    void drawFrameFrustum(const Pose3D &pose, const CameraModel &camera, bool useTex, unsigned int texTarget=0, unsigned int texID=0);
};

} /* namespace dtslam */

#endif /* MAPWINDOW_H_ */
