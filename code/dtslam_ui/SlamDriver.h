/*
 * SlamDriver.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef SLAMDRIVER_H_
#define SLAMDRIVER_H_

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>

#include "Application.h"

#include "dtslam/FeatureIndexer.h"
#include "dtslam/SlamSystem.h"

#include "shaders/DTSlamShaders.h"
#include "windows/BaseWindow.h"

namespace dtslam
{
class ImageDataSource;
class OpenCVDataSource;

class SlamDriver: public Application
{
private:
	bool mInitialized;

    int mFrameCount;

    bool mUsingCamera;
    std::unique_ptr<ImageDataSource> mImageSrc;
    int mDownsampleInputCount;
    cv::Size2i mImageSize;

    DTSlamShaders mShaders;

    volatile bool mQuit;

    bool mFrameByFrame;
    bool mAdvanceFrame;

    bool mRecordFrames;
    int mRecordId;
    std::string mRecordFileFormat;

    bool mShowProfiler;
    bool mShowProfilerTotals;
	
	float mFPS;
	std::chrono::high_resolution_clock::time_point mLastFPSCheck;
	std::chrono::high_resolution_clock::duration mFPSUpdateDuration;
	std::chrono::high_resolution_clock::duration mFPSSampleAccum;
	int mFPSSampleCount;

    std::unique_ptr<CameraModel> mCamera;
    SlamSystem mSlam;

    std::vector<std::unique_ptr<BaseWindow>> mWindows;
    BaseWindow *mActiveWindow;

    cv::Point3f mARCubeCenter;
    cv::Point3f mARCubeAxes[3];

public:
    SlamDriver();
    ~SlamDriver();

    bool getFinished() {return mQuit;}

    DTSlamShaders &getShaders() {return mShaders;}

    bool isARCubeValid() const {return mARCubeAxes[0].x != 0 || mARCubeAxes[0].y != 0 || mARCubeAxes[0].z != 0;}
    const cv::Point3f &getARCubeCenter() const {return mARCubeCenter;}
    const cv::Point3f &getARCubeAxis(int i) const {return mARCubeAxes[i];}
    void setARCube(const cv::Point3f &center, const cv::Point3f &axis1, const cv::Point3f &axis2, const cv::Point3f &axis3)
    {
    	mARCubeCenter = center;
    	mARCubeAxes[0]=axis1;
    	mARCubeAxes[1]=axis2;
    	mARCubeAxes[2]=axis3;
    }
    void disableARCube() {mARCubeCenter = cv::Point3f(0,0,0); mARCubeAxes[0] = cv::Point3f(0,0,0);}
    void generateARCubeVertices(std::vector<unsigned int> &triangleIndices, std::vector<cv::Vec4f> &vertices, std::vector<cv::Vec4f> &colors, std::vector<cv::Vec3f> &normals);

    bool init();
    void resize();
    void exit();
    bool loop() {return mQuit;}

    void keyDown(bool isSpecial, uchar key);
    void keyUp(bool isSpecial, uchar key);
    void touchDown(int id, int x, int y);
    void touchMove(int x, int y);
    void touchUp(int id, int x, int y);

    void draw(void);

private:
    bool initImageSrc();

    KeyBindingHandler<SlamDriver> mKeyBindings;
    void runVideo() {mFrameByFrame = !mFrameByFrame; mAdvanceFrame = false;}
    void stepVideo(){mFrameByFrame = true; mAdvanceFrame = true;}
    void toggleProfilerMode();
    void resetProfiler() {Profiler::Instance().reset();}
    void escapePressed() {mQuit=true;}
    void changeWindowKey(bool isSpecial, unsigned char key);
    void resetSystem();
    void startRecording();
    void recordFrame(cv::Mat3b &im);
	void resyncTracker();

	void saveMap();
	void loadMap();

    void setActiveWindow(BaseWindow *window);
};

}

#endif /* SLAMDRIVER_H_ */
