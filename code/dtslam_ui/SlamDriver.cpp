/*
 * SlamDriver.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "SlamDriver.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <GL/freeglut.h>

#include <ceres/loss_function.h>

#undef LOG
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <glog/logging.h>
#include "dtslam/log.h"

#include "ImageDataSource.h"

#include "dtslam/Profiler.h"
#include "dtslam/flags.h"
#include "dtslam/Serializer.h"

#include "OpenCVDataSource.h"
#include "SequenceDataSource.h"
#include "UserInterfaceInfo.h"
#include "ViewportTiler.h"

#include "windows/MatchesWindow.h"
#include "windows/MapExpanderWindow.h"
#include "windows/MapWindow.h"
#include "windows/KeyFramePairWindow.h"
#include "windows/ARWindow.h"
#include "windows/TestMatchWindow.h"
#include "windows/FrameLinkWindow.h"

#include "flags.h"

namespace dtslam
{
SlamDriver::SlamDriver()
        : mInitialized(false), mFrameCount(0), mQuit(false),
          mFrameByFrame(true), mAdvanceFrame(false), mShowProfiler(true), mShowProfilerTotals(false),
          mActiveWindow(NULL), mKeyBindings(this),
		  mRecordFrames(false)

{
}

SlamDriver::~SlamDriver()
{
}

// Initialization for this application
bool SlamDriver::init(void)
{
    std::cout << "SlamDriver init." << std::endl;
	
#ifdef ENABLE_LOG
	std::cout << "Logging is enabled. This severely hits performance! It can be disabled in dtslam/log.h." << std::endl;
#else
	std::cout << "Logging is disabled. See dtslam/log.h to enable it." << std::endl;
#endif

#ifdef ENABLE_PROFILER
	std::cout << "Profiling is enabled.\n";
#else
	std::cout << "Profiling is disabled. See dtslam/Profiler.h to enable it.\n";
#endif

    Profiler::Instance().setCurrentThreadName("render");
    char glogstr[] = "dtslam";

    google::InitGoogleLogging(glogstr);

    if (!initImageSrc())
    {
    	DTSLAM_LOG << "Couldn't initialize image source.\n";
        return false;
    }

    // Initialize Shader
    if (!mShaders.init())
    {
		DTSLAM_LOG << "Couldn't initialize shaders.\n";
		return false;
    }

	//Determine downscale at input
	int width = mImageSrc->getSourceSize().width;
	mDownsampleInputCount = 0;
	while(width > FLAGS_DriverMaxImageWidth)
	{
		width = (width+1)/2;
		mDownsampleInputCount++;
	}
	int scale = 1<<mDownsampleInputCount;

	mImageSrc->setDownsample(mDownsampleInputCount);
	mImageSize = mImageSrc->getSize();
	DTSLAM_LOG << "Input image size after downsampling: " << mImageSize << "\n";

	//Check size vs calibration
	cv::Size expectedSize(FLAGS_CameraWidth/scale, FLAGS_CameraHeight/scale);
	if(mImageSize.width != expectedSize.width || mImageSize.height != expectedSize.height)
	{
		DTSLAM_LOG << "Warning: image size " << mImageSize << " does not match calibration size " << expectedSize << "\n";
	}

	//Get first frame
	if(!mImageSrc->update())
    {
    	DTSLAM_LOG << "Couldn't get first frame from image source.\n";
    	return false;
    }

    //Init camera
	mCamera.reset(new CameraModel());
	mCamera->init((float)FLAGS_CameraFx / scale, (float)FLAGS_CameraFy / scale, (float)FLAGS_CameraU0 / scale, (float)FLAGS_CameraV0 / scale, FLAGS_CameraWidth / scale, FLAGS_CameraHeight / scale);
	mCamera->getDistortionModel().init((float)FLAGS_CameraK1, (float)FLAGS_CameraK2);
	mCamera->getDistortionModel().setMaxRadius(mCamera->getMaxRadiusSq(mImageSize));
	mCamera->initLUT();

	//Slam system
	cv::Mat1b imageGray = mImageSrc->getImgGray();
	cv::Mat3b imageColor = mImageSrc->getImgColor();
	mSlam.init(mCamera.get(), mImageSrc->getCaptureTime(), imageColor, imageGray);
	mSlam.setSingleThreaded(FLAGS_DriverSingleThreaded);

	//Add windows
	mWindows.push_back(std::unique_ptr<BaseWindow>(new MatchesWindow()));
	mWindows.push_back(std::unique_ptr<BaseWindow>(new MapExpanderWindow()));
	mWindows.push_back(std::unique_ptr<BaseWindow>(new MapWindow()));
	mWindows.push_back(std::unique_ptr<BaseWindow>(new KeyFramePairWindow()));
	mWindows.push_back(std::unique_ptr<BaseWindow>(new ARWindow()));
	mWindows.push_back(std::unique_ptr<BaseWindow>(new TestMatchWindow()));
	mWindows.push_back(std::unique_ptr<BaseWindow>(new FrameLinkWindow()));

	//Add bindings
	mKeyBindings.addBinding(true,GLUT_KEY_F5,static_cast<KeyBindingHandler<SlamDriver>::SimpleBindingFunc>(&SlamDriver::runVideo),"Run the video stream.");
	mKeyBindings.addBinding(true, GLUT_KEY_F8, static_cast<KeyBindingHandler<SlamDriver>::SimpleBindingFunc>(&SlamDriver::saveMap), "Save the map to disk.");
	mKeyBindings.addBinding(true, GLUT_KEY_F9, static_cast<KeyBindingHandler<SlamDriver>::SimpleBindingFunc>(&SlamDriver::loadMap), "Load the map from disk.");
	mKeyBindings.addBinding(false, ' ', static_cast<KeyBindingHandler<SlamDriver>::SimpleBindingFunc>(&SlamDriver::stepVideo), "Advance one frame.");
	mKeyBindings.addBinding(false,'p',static_cast<KeyBindingHandler<SlamDriver>::SimpleBindingFunc>(&SlamDriver::toggleProfilerMode),"Toggle profiler mode.");
	mKeyBindings.addBinding(false,'P',static_cast<KeyBindingHandler<SlamDriver>::SimpleBindingFunc>(&SlamDriver::resetProfiler),"Reset profiler counts.");
	mKeyBindings.addBinding(false,'r',static_cast<KeyBindingHandler<SlamDriver>::SimpleBindingFunc>(&SlamDriver::resetSystem),"Reset the slam system.");
	mKeyBindings.addBinding(false,'R',static_cast<KeyBindingHandler<SlamDriver>::SimpleBindingFunc>(&SlamDriver::startRecording),"Reset and start recording.");
	mKeyBindings.addBinding(true,GLUT_KEY_F1,static_cast<KeyBindingHandler<SlamDriver>::SimpleBindingFunc>(&SlamDriver::resyncTracker),"Resyncs the tracker with/without 2D matches.");

	for(int i=0; i<(int)mWindows.size(); ++i)
		mKeyBindings.addBinding(false,i+'1',static_cast<KeyBindingHandler<SlamDriver>::BindingFunc>(&SlamDriver::changeWindowKey),"Show window: " + mWindows[i]->getName());

	mKeyBindings.addBinding(false,27,static_cast<KeyBindingHandler<SlamDriver>::SimpleBindingFunc>(&SlamDriver::escapePressed),"Quit.");

	DTSLAM_LOG << "\nBasic keys:\n";
	mKeyBindings.showHelp();

	setActiveWindow(mWindows[0].get());

	disableARCube();

    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_LINE_SMOOTH);

	mFPS = 0;
	mLastFPSCheck = std::chrono::high_resolution_clock::now();
	mFPSUpdateDuration = std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(std::chrono::seconds(1));
	mFPSSampleAccum = std::chrono::high_resolution_clock::duration(0);
	mFPSSampleCount = 0;

	mInitialized = true;
    return true;
}

void SlamDriver::resetSystem()
{
	cv::Mat1b imageGray = mImageSrc->getImgGray();
	cv::Mat3b imageColor = mImageSrc->getImgColor();
	mSlam.init(mCamera.get(), mImageSrc->getCaptureTime(), imageColor, imageGray);

	for(int i=0; i<(int)mWindows.size(); ++i)
		mWindows[i]->requireInit();

	if(mActiveWindow)
	{
		mActiveWindow->init(this, &mSlam, mImageSize);
		mActiveWindow->updateState();
	}
}

void SlamDriver::startRecording()
{
	resetSystem();
	mRecordFrames = true;
	mRecordId = 0;
	mRecordFileFormat = FLAGS_DriverRecordPath + "frame%.4d.jpg";

	//Delete all previous
	bool filesToDelete=true;
	int deleteId=0;
	char buffer[1024];
	while(filesToDelete)
	{
		sprintf(buffer, mRecordFileFormat.c_str(), deleteId++);
		if(std::remove(buffer))
			filesToDelete = false;
	};
}

void SlamDriver::recordFrame(cv::Mat3b &im)
{
	//Convert to bgr
	cv::Mat3b bgr;
	cv::cvtColor(im, bgr, cv::COLOR_RGB2BGR);

	//Save
	char buffer[1024];
	sprintf(buffer, mRecordFileFormat.c_str(), mRecordId++);

	cv::imwrite(buffer, bgr);
}

void SlamDriver::setActiveWindow(BaseWindow *window)
{
	if(!window->isInitialized())
	{
		window->init(this, &mSlam, mImageSize);
		window->setCurrentImageTexture(mImageSrc->getTextureTarget(), mImageSrc->getTextureId());
		window->showHelp();
	}
	else
	{
		window->resize();
	}
	window->updateState();
	mActiveWindow = window;
}

void SlamDriver::resize()
{
	mActiveWindow->resize();
}

bool SlamDriver::initImageSrc()
{
	mUsingCamera = false;

	if (FLAGS_DriverCameraId >= 0)
	{
		//Use camera
		OpenCVDataSource *source = new OpenCVDataSource();
		mImageSrc.reset(source);
		if (!source->open(FLAGS_DriverCameraId))
		{
			DTSLAM_LOG << "Error opening camera.\n";
			return false;
		}

		mUsingCamera = true;
		DTSLAM_LOG << "Camera opened succesfully\n";
	}
	else if (!FLAGS_DriverSequenceFormat.empty())
	{
		//Use image sequence
		std::string sequence = FLAGS_DriverDataPath + "/" + FLAGS_DriverSequenceFormat;

		DTSLAM_LOG << "Image sequence: " << sequence << "\n";
		SequenceDataSource *source = new SequenceDataSource();
		mImageSrc.reset(source);
		if (!source->open(sequence, FLAGS_DriverSequenceStartIdx))
		{
			DTSLAM_LOG << "Error opening sequence.\n";
			return false;
		}

		DTSLAM_LOG << "Opened image sequence succesfully\n";
	}
	else if (!FLAGS_DriverVideoFile.empty())
	{
		//Use video file
	    std::string videoFilename = FLAGS_DriverDataPath + "/" + FLAGS_DriverVideoFile;

	    DTSLAM_LOG << "Video file: " << videoFilename << "\n";
		OpenCVDataSource *source = new OpenCVDataSource();
	    mImageSrc.reset(source);
	    if(!source->open(videoFilename))
	    {
	    	DTSLAM_LOG << "Error opening video.\n";
	        return false;
	    }

	    DTSLAM_LOG << "Opened video file succesfully\n";
	}
	else
	{
		DTSLAM_LOG << "No image source specified. Set either DriverCameraId, DriverVideoFile, or DriverSequenceFormat.\n";
		return false;
	}

	DTSLAM_LOG << "Image source size: " << mImageSrc->getSourceSize() << "\n";

	return true;
}

void SlamDriver::exit()
{
    DTSLAM_LOG << "clean exit...\n";
    mShaders.free();
}

void SlamDriver::keyDown(bool isSpecial, uchar key)
{
	if(mKeyBindings.dispatchKeyDown(isSpecial, key))
		return;
	mActiveWindow->keyDown(isSpecial, key);
}

void SlamDriver::keyUp(bool isSpecial, uchar key)
{
	if(mKeyBindings.dispatchKeyUp(isSpecial, key))
		return;
	mActiveWindow->keyUp(isSpecial, key);
}

void SlamDriver::toggleProfilerMode()
{
	if(!mShowProfiler)
		mShowProfiler = true;
	else if(!mShowProfilerTotals)
		mShowProfilerTotals = true;
	else
		mShowProfiler = mShowProfilerTotals = false;
Profiler::Instance().setShowTotals(mShowProfilerTotals);
}

void SlamDriver::changeWindowKey(bool isSpecial, unsigned char key)
{
	int idx = (key-'0')-1;
	if(idx < 0)
		idx = 10;
	if(idx >= (int)mWindows.size())
		idx = mWindows.size()-1;
	setActiveWindow(mWindows[idx].get());
}

void SlamDriver::touchDown(int id, int x, int y)
{
	mActiveWindow->touchDown(id, x, y);
}

void SlamDriver::touchMove(int x, int y)
{
	mActiveWindow->touchMove(x, y);
}

void SlamDriver::touchUp(int id, int x, int y)
{
	mActiveWindow->touchUp(id, x, y);
}

// Main draw function
void SlamDriver::draw(void)
{
    if(!mFrameByFrame || mAdvanceFrame)
	{
        bool isFrameAvailable;
        {
        	ProfileSection section("updateFrame");

        	//Drop frames
        	mImageSrc->dropFrames(FLAGS_DriverDropFrames);

			isFrameAvailable = mImageSrc->update();
        }
        if(isFrameAvailable)
		{
            ProfileSection section("execute");

			mFrameCount++;
			mAdvanceFrame = false;

			if (!mUsingCamera)
				DTSLAM_LOG << "\nFrame #" << mFrameCount << "\n";

			//Read new input frame
			cv::Mat1b imageGray = mImageSrc->getImgGray();
			cv::Mat3b imageColor = mImageSrc->getImgColor();

			//Record
			if(mRecordFrames)
				recordFrame(imageColor);

			//Process new frame
			auto tic = std::chrono::high_resolution_clock::now();
			mSlam.processImage(mImageSrc->getCaptureTime(), imageColor, imageGray);
			mFPSSampleAccum += std::chrono::high_resolution_clock::now()-tic;
			mFPSSampleCount++;

			mActiveWindow->updateState();
		}
	}

	mSlam.idle();

    {
        ProfileSection section("draw");

		//glClearColor(1.0, 1.0, 1.0, 1.0);
		glClearColor(0.0, 0.0, 0.0, 0.0);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        {
        	ProfileSection section("windowDraw");
        	mActiveWindow->draw();
        }

	    //Text
	    std::stringstream ss;
		ss << "FPS " << (int)mFPS << "\n";
		ss << "Frame " << mFrameCount << "\n";

	    if(mShowProfiler)
		{
		    Profiler::Instance().logStats(ss);
		}

	    cv::Size2i screenSize = UserInterfaceInfo::Instance().getScreenSize();
		float viewportAspect = static_cast<float>(screenSize.width) / screenSize.height;
		glViewport(0,0,screenSize.width,screenSize.height);
		mShaders.getText().setMVPMatrix(ViewportTiler::GetImageSpaceMvp(viewportAspect, screenSize));

	    mShaders.getText().setActiveFontSmall();
	    mShaders.getText().setRenderCharHeight(10);
	    mShaders.getText().setCaret(cv::Point2f(0,0));
	    mShaders.getText().setColor(StaticColors::Green());
	    mShaders.getText().renderText(ss);

	    //Map stats in the bottom
		const float kMapStatsFontHeight = 10.0f;
		cv::Point2f corners[] = { cv::Point2f(0.0f, (float)screenSize.height - 2 * kMapStatsFontHeight), cv::Point2f((float)screenSize.width, (float)screenSize.height - 2 * kMapStatsFontHeight),
			cv::Point2f(0.0f, (float)screenSize.height), cv::Point2f((float)screenSize.width, (float)screenSize.height) };
	    mShaders.getColor().setMVPMatrix(ViewportTiler::GetImageSpaceMvp(viewportAspect, screenSize));
	    mShaders.getColor().drawVertices(GL_TRIANGLE_STRIP, corners, 4, StaticColors::Black(0.5f));

		mShaders.getText().setRenderCharHeight(kMapStatsFontHeight);
		mShaders.getText().setCaret(corners[0] + cv::Point2f(kMapStatsFontHeight/2, kMapStatsFontHeight/2));
		mShaders.getText().setColor(StaticColors::White());
		{
			TextRendererStream ts(mShaders.getText());

			int frameCount = mSlam.getMap().getTotalFrameCount();
			int count3D = mSlam.getMap().getTotalFeature3DCount();
			int count2D = mSlam.getMap().getTotalFeature2DCount();
			if (!FLAGS_DisableRegions)
				ts << "Regions: " << mSlam.getMap().getRegions().size() << ", ";
			ts << "Keyframes: " << frameCount << ", Features (2D: " << count2D << ", 3D : " << count3D << ")";

			switch (mSlam.getTracker().getPoseType() )
			{
				case EPoseEstimationType::PureRotation:
					ts.setColor(StaticColors::Yellow());
					ts << " PURE ROTATION";
					break;
				case EPoseEstimationType::Essential:
					ts.setColor(StaticColors::Green());
					ts << " ESSENTIAL MODEL";
					break;
				case EPoseEstimationType::Invalid:
					ts.setColor(StaticColors::Red());
					ts << " LOST";
					break;
			}
			ts.setColor(StaticColors::White());

			//Expander status
			switch (mSlam.getMapExpander().getStatus())
			{
			case ESlamMapExpanderStatus::CheckingFrame:
				ts << ", expander checking";
				break;
			case ESlamMapExpanderStatus::AddingFrame:
				ts << ", expander adding";
				break;
			case ESlamMapExpanderStatus::SingleFrameBA:
				ts << ", expander one-frame BA";
				break;
			}

			//BA status
			if (mSlam.isBARunning())
			{
				ts << ", ";
				if (mSlam.getActiveRegion()->getAbortBA())
				{
					ts.setColor(StaticColors::Yellow());
					ts << "aborting BA";
					ts.setColor(StaticColors::White());
				}
				else
				{
					ts << "BA is running";
				}
			}
			else if (mSlam.getActiveRegion()->getShouldBundleAdjust())
			{
				ts << ", ";
				ts.setColor(StaticColors::Yellow());
				ts << "BA pending";
				ts.setColor(StaticColors::White());
			}
		}
    }

	//Update FPS
	auto now = std::chrono::high_resolution_clock::now();
	auto elapsedDuration = now - mLastFPSCheck;
	if (elapsedDuration > mFPSUpdateDuration)
	{
		if (mFPSSampleCount)
			mFPS = mFPSSampleCount / std::chrono::duration_cast<std::chrono::duration<float>>(mFPSSampleAccum).count();
		
		mFPSSampleCount = 0;
		mFPSSampleAccum = std::chrono::high_resolution_clock::duration(0);

		mLastFPSCheck = now;
	}
}

void SlamDriver::generateARCubeVertices(std::vector<unsigned int> &triangleIndices, std::vector<cv::Vec4f> &vertices, std::vector<cv::Vec4f> &colors, std::vector<cv::Vec3f> &normals)
{
    const cv::Vec4f cubeCorners[] = {cvutils::PointToHomogenous(mARCubeCenter-mARCubeAxes[0]-mARCubeAxes[1]-mARCubeAxes[2]),	//0 ---
    								cvutils::PointToHomogenous(mARCubeCenter+mARCubeAxes[0]-mARCubeAxes[1]-mARCubeAxes[2]),		//1 +--
    								cvutils::PointToHomogenous(mARCubeCenter+mARCubeAxes[0]+mARCubeAxes[1]-mARCubeAxes[2]),		//2 ++-
    								cvutils::PointToHomogenous(mARCubeCenter-mARCubeAxes[0]+mARCubeAxes[1]-mARCubeAxes[2]),		//3 -+-
    								cvutils::PointToHomogenous(mARCubeCenter-mARCubeAxes[0]-mARCubeAxes[1]+mARCubeAxes[2]),		//4 --+
    								cvutils::PointToHomogenous(mARCubeCenter+mARCubeAxes[0]-mARCubeAxes[1]+mARCubeAxes[2]),		//5 +-+
    								cvutils::PointToHomogenous(mARCubeCenter+mARCubeAxes[0]+mARCubeAxes[1]+mARCubeAxes[2]),		//6 +++
    								cvutils::PointToHomogenous(mARCubeCenter-mARCubeAxes[0]+mARCubeAxes[1]+mARCubeAxes[2])}; 	//7 -++

    vertices.resize(6 * 4); //Six faces, four vertices per face
    colors.resize(6 * 4);
    normals.resize(6 * 4);
    triangleIndices.resize(6*6); //Six faces, six indices per face

    int faceIdx;

    //Face down
    faceIdx = 0;
    vertices[faceIdx*4 + 0] = cubeCorners[0]; //---
    vertices[faceIdx*4 + 1] = cubeCorners[1]; //+--
    vertices[faceIdx*4 + 2] = cubeCorners[2]; //++-
    vertices[faceIdx*4 + 3] = cubeCorners[3]; //-+-

    normals[faceIdx*4 + 0] = normals[faceIdx*4 + 1] = normals[faceIdx*4 + 2] = normals[faceIdx*4 + 3] = -mARCubeAxes[2];

    //Face up
    faceIdx = 1;
    vertices[faceIdx*4 + 0] = cubeCorners[4]; //--+
    vertices[faceIdx*4 + 1] = cubeCorners[5]; //+-+
    vertices[faceIdx*4 + 2] = cubeCorners[6]; //+++
    vertices[faceIdx*4 + 3] = cubeCorners[7]; //-++

    normals[faceIdx*4 + 0] = normals[faceIdx*4 + 1] = normals[faceIdx*4 + 2] = normals[faceIdx*4 + 3] = mARCubeAxes[2];

    //Face right
    faceIdx = 2;
    vertices[faceIdx*4 + 0] = cubeCorners[5]; //+-+
    vertices[faceIdx*4 + 1] = cubeCorners[6]; //+++
    vertices[faceIdx*4 + 2] = cubeCorners[2]; //++-
    vertices[faceIdx*4 + 3] = cubeCorners[1]; //+--

    normals[faceIdx*4 + 0] = normals[faceIdx*4 + 1] = normals[faceIdx*4 + 2] = normals[faceIdx*4 + 3] = mARCubeAxes[0];

    //Face left
    faceIdx = 3;
    vertices[faceIdx*4 + 0] = cubeCorners[4]; //--+
    vertices[faceIdx*4 + 1] = cubeCorners[7]; //-++
    vertices[faceIdx*4 + 2] = cubeCorners[3]; //-+-
    vertices[faceIdx*4 + 3] = cubeCorners[0]; //---

    normals[faceIdx*4 + 0] = normals[faceIdx*4 + 1] = normals[faceIdx*4 + 2] = normals[faceIdx*4 + 3] = -mARCubeAxes[0];

    //Face front
    faceIdx = 4;
    vertices[faceIdx*4 + 0] = cubeCorners[7]; //-++
    vertices[faceIdx*4 + 1] = cubeCorners[6]; //+++
    vertices[faceIdx*4 + 2] = cubeCorners[2]; //++-
    vertices[faceIdx*4 + 3] = cubeCorners[3]; //-+-

    normals[faceIdx*4 + 0] = normals[faceIdx*4 + 1] = normals[faceIdx*4 + 2] = normals[faceIdx*4 + 3] = mARCubeAxes[1];

    //Face back
    faceIdx = 5;
    vertices[faceIdx*4 + 0] = cubeCorners[5]; //+-+
    vertices[faceIdx*4 + 1] = cubeCorners[4]; //--+
    vertices[faceIdx*4 + 2] = cubeCorners[0]; //---
    vertices[faceIdx*4 + 3] = cubeCorners[1]; //+--

    normals[faceIdx*4 + 0] = normals[faceIdx*4 + 1] = normals[faceIdx*4 + 2] = normals[faceIdx*4 + 3] = -mARCubeAxes[1];

    //Other properties
    for(faceIdx=0; faceIdx<6; faceIdx++)
    {
    	//Set color
        colors[faceIdx*4 + 0] = colors[faceIdx*4 + 1] = colors[faceIdx*4 + 2] = colors[faceIdx*4 + 3] = StaticColors::Green(1, 0.5f + 0.5f/5*faceIdx);

        //Set indices
        triangleIndices[faceIdx*6 + 0] = faceIdx*4 + 0;
        triangleIndices[faceIdx*6 + 1] = faceIdx*4 + 1;
        triangleIndices[faceIdx*6 + 2] = faceIdx*4 + 2;
        triangleIndices[faceIdx*6 + 3] = faceIdx*4 + 0;
        triangleIndices[faceIdx*6 + 4] = faceIdx*4 + 2;
        triangleIndices[faceIdx*6 + 5] = faceIdx*4 + 3;
    }
}

void SlamDriver::resyncTracker()
{
	//This function alternates between using 2D features and not, resyncing the tracker after, so that the effect of the 2D featues can be clearly observed.

	shared_lock_guard<shared_mutex> lock(mSlam.getMap().getMutex());

	FLAGS_PoseUse2D = !FLAGS_PoseUse2D;

	DTSLAM_LOG << "\nResyncing, PoseUse2D=" << FLAGS_PoseUse2D << "\n";

	mSlam.getTracker().resync();

	mActiveWindow->updateState();

	DTSLAM_LOG << "\nResyncing, PoseUse2D=" << FLAGS_PoseUse2D << ", done\n";

	ceres::CauchyLoss loss(3);
	double robustError[3];

	auto &matches = mSlam.getTracker().getMatches();
	auto &errors = mSlam.getTracker().getReprojectionErrors();

	float total3D = 0;
	float total2D = 0;
	for (int i = 0, end = matches.size(); i != end; ++i)
	{
		auto &match = matches[i];
		auto &err = errors[i];

		loss.Evaluate(err.bestReprojectionErrorSq, robustError);

		if (match.measurement.getFeature().is3D())
			total3D += (float)robustError[0];
		else
			total2D += (float)robustError[0];
	}

	DTSLAM_LOG << "3D total error=" << total3D << "\n";
	DTSLAM_LOG << "2D total error=" << total2D << "\n";
	DTSLAM_LOG << "Total error=" << total3D+total2D << "\n";
}

void SlamDriver::saveMap()
{
	Serializer s;
	s.open("save/", "map.yml");
	s.addObject(&mSlam.getMap());
	s.serializeAll();
}

void SlamDriver::loadMap()
{
	Deserializer s;
	s.open("save/", "map.yml");
	s.deserialize();

	mCamera = s.getObjectForOwner<CameraModel>();
	std::unique_ptr<SlamMap> map = s.getObjectForOwner<SlamMap>();
	mSlam.init(mCamera.get(), std::move(map));
}

}
