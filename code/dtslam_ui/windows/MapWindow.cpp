/*
 * MapWindow.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "MapWindow.h"
#include <thread>
#include <array>
#include "../shaders/DTSlamShaders.h"
#include "dtslam/SlamSystem.h"
#include "dtslam/SlamMap.h"
#include "dtslam/PoseTracker.h"
#include "dtslam/BundleAdjuster.h"
#include "dtslam/flags.h"
#include "../flags.h"
#include "../SlamDriver.h"

namespace dtslam {

void MapWindow::showHelp() const
{
	BaseWindow::showHelp();
	DTSLAM_LOG << "Use left mouse button to rotate view and right mouse button to translate view.\n"
			<< "Green squares = 2D features\n"
			<< "Blue squares = 3D features\n"
			<< "Purple squares = temporary triangulations from tracker\n";
}

bool MapWindow::init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize)
{
	BaseWindow::init(app, slam, imageSize);
	
	mMapDrawScale = (float)FLAGS_MapDrawScale;

	mMap = &mSlam->getMap();
	mRegion = mMap->getRegions()[0].get();
	mTracker = &mSlam->getTracker();

	mViewerPose.set(cv::Matx33f::eye(), cv::Vec3f(0,0,3));

	resize();

	mKeyBindings.addBinding(false,'b',static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&MapWindow::performBA),"Perform Bundle Adjustment on demand.");
	mKeyBindings.addBinding(false,'f',static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&MapWindow::forceNewKeyFrame),"Force a new key frame to be added.");
	mKeyBindings.addBinding(false,'t',static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&MapWindow::togglePatches),"Toggle between patch and point display.");
	//mKeyBindings.addBinding(false,'+',static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&MapWindow::increasePointSize),"Increase GL point size.");
	//mKeyBindings.addBinding(false,'-',static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&MapWindow::decreasePointSize),"Decrease GL point size.");
	mKeyBindings.addBinding(false, '-', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&MapWindow::zoomIn), "Zoom in.");
	mKeyBindings.addBinding(false, '+', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&MapWindow::zoomOut), "Zoom out.");
	mKeyBindings.addBinding(false, 'c', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&MapWindow::startCube), "Define cube surface.");
	mKeyBindings.addBinding(true,GLUT_KEY_F2,static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&MapWindow::selectNextRegion),"Select next region.");
		
	return true;
}

void MapWindow::updateState()
{
	shared_lock<shared_mutex> lockRead(mSlam->getMap().getMutex());

	ensureValidRegion();

	//Determine features in view
    //Add 3D features
	mFeaturesInView.clear();
	for(auto itF=mRegion->getFeatures3D().begin(),endF=mRegion->getFeatures3D().end(); itF!=endF; ++itF)
	{
		const SlamFeature &feature = **itF;

		//Project
		const cv::Point3f xc = mViewerPose.apply(feature.getPosition());

		//Inside image frame?
		const cv::Point2f pos = mViewerCamera.projectFromWorld(xc);
		if(!mViewerCamera.isPointInside(xc, pos))
			continue;

		//Add
		mFeaturesInView.push_back(&feature);
	}

	//Add 2D features
	for(auto itF=mRegion->getFeatures2D().begin(),endF=mRegion->getFeatures2D().end(); itF!=endF; ++itF)
	{
		SlamFeature &feature = **itF;

		//Restrict octave
		const SlamFeatureMeasurement &m= *feature.getMeasurements()[0];

		//Project
		const Pose3D &refPose = m.getKeyFrame().getPose();
		const cv::Point3f center = refPose.applyInv(m.getUniquePositionXn()*mMapDrawScale);
		const cv::Point3f xc = mViewerPose.apply(center);

		//Inside image frame?
		cv::Point2f pos = mViewerCamera.projectFromWorld(xc);
		if(!mViewerCamera.isPointInside(xc, pos))
			continue;

		//Add
		mFeaturesInView.push_back(&feature);
	}


    /////////////////////////
    // Prepare draw objects
    mFeaturesToDraw.clear();
    mMatchedFeaturesToDraw.clear();
    mFrustumsToDraw.clear();

	//Pose log
	mPoseLog.clear();
	//for (auto &log : mSlam->mPoseLog)
	//{
	//	mPoseLog.push_back(cvutils::PointToHomogenous(log.second + log.first->getPose().getCenter()));
	//}
	//for (auto &keyframe : mRegion->getKeyFrames())
	//{
	//	mPoseLog.push_back(cvutils::PointToHomogenous(keyframe->getPose().getCenter()));
	//}

    //Features
	for(auto featurePtr : mFeaturesInView)
	{
		const SlamFeature &feature = *featurePtr;

		DrawFeatureData data;
		bool isMatched = isFeatureMatched(feature);

		cv::Point3f center;

		//Color
		const float alpha = 0.55f;
		
		//Black color scheme
		//switch (feature.getStatus())
		//{
		//case SlamFeatureStatus::NotTriangulated: data.solidColor = StaticColors::Yellow(alpha); break;
		//case SlamFeatureStatus::TwoViewTriangulation: data.solidColor = StaticColors::Cyan(alpha); break;
		//case SlamFeatureStatus::ThreeViewAgreement: data.solidColor = StaticColors::Blue(alpha); break;
		//case SlamFeatureStatus::ThreeViewDisagreement: data.solidColor = StaticColors::Red(alpha); break;
		//case SlamFeatureStatus::MultiViewAgreement: data.solidColor = StaticColors::White(alpha); break;
		//case SlamFeatureStatus::MultiViewDisagreement: data.solidColor = StaticColors::Purple(alpha); break;			
		//}

		//White color scheme
		//switch (feature.getStatus())
		//{
		//case SlamFeatureStatus::NotTriangulated: data.solidColor = StaticColors::Green(alpha); break;
		//case SlamFeatureStatus::TwoViewTriangulation: 
		//	//data.solidColor = StaticColors::Cyan(alpha); break;
		//case SlamFeatureStatus::ThreeViewAgreement:
		//case SlamFeatureStatus::MultiViewAgreement: 
		//	data.solidColor = StaticColors::Blue(alpha); break;
		//case SlamFeatureStatus::ThreeViewDisagreement:
		//	//data.solidColor = StaticColors::Cyan(0.1f); break;
		//case SlamFeatureStatus::MultiViewDisagreement:
		//	data.solidColor = StaticColors::Blue(alpha); break;
		//}

		//Region color scheme
		if (feature.getStatus() == SlamFeatureStatus::NotTriangulated)
			data.solidColor = StaticColors::Green();
		else
			data.solidColor = mRegionColors[feature.mOriginalRegionId % mRegionColors.size()];
		data.solidColor[3] = alpha;

		//Position
		cv::Point3f axis1;
		if (feature.is3D())
		{
			center = feature.getPosition();
			axis1 = feature.getPlusOneOffset();
		}
		else
		{
    		const SlamFeatureMeasurement &m= *feature.getMeasurements()[0];
			const Pose3D &refPose = m.getKeyFrame().getPose();
			const cv::Point3f refPosition = m.getUniquePositionXn();
			center = refPose.applyInv(refPosition*mMapDrawScale);
			axis1 = feature.getPlusOneOffset()*mMapDrawScale;
		}

		//Check to see if feature has been matched
		if(isMatched)
			data.solidColor = StaticColors::Yellow(0.55f);

		cv::Point3f axis2 = -cv::Vec3f(axis1).cross(feature.getNormal());

		data.vertices[0] = cvutils::PointToHomogenous(center-PatchWarper::kPatchCenterOffset*axis1-PatchWarper::kPatchCenterOffset*axis2);
		data.vertices[1] = cvutils::PointToHomogenous(center-PatchWarper::kPatchCenterOffset*axis1+PatchWarper::kPatchRightSize*axis2);
		data.vertices[2] = cvutils::PointToHomogenous(center+PatchWarper::kPatchRightSize*axis1-PatchWarper::kPatchCenterOffset*axis2);
		data.vertices[3] = cvutils::PointToHomogenous(center+PatchWarper::kPatchRightSize*axis1+PatchWarper::kPatchRightSize*axis2);
		data.center = cvutils::PointToHomogenous(center);

		const SlamFeatureMeasurement &measurement = *feature.getMeasurements()[0];
		const int scale = 1<<measurement.getOctave();
		const TextureHelper &frameTexture = getFrameTexture(measurement.getKeyFrame());

		const int maxWidth = frameTexture.getSize().width-1;
		const int maxHeight = frameTexture.getSize().height-1;

		data.useTex = true;
		data.texTarget = GL_TEXTURE_2D;
		data.texId = frameTexture.getId();

		auto pos = measurement.getUniquePosition();
		data.texCoordinates[0][0] = (pos.x - scale*PatchWarper::kPatchCenterOffset) / maxWidth;
		data.texCoordinates[0][1] = (pos.y - scale*PatchWarper::kPatchCenterOffset) / maxHeight;

		data.texCoordinates[1][0] = (pos.x - scale*PatchWarper::kPatchCenterOffset) / maxWidth;
		data.texCoordinates[1][1] = (pos.y + scale*PatchWarper::kPatchRightSize) / maxHeight;

		data.texCoordinates[2][0] = (pos.x + scale*PatchWarper::kPatchRightSize) / maxWidth;
		data.texCoordinates[2][1] = (pos.y - scale*PatchWarper::kPatchCenterOffset) / maxHeight;

		data.texCoordinates[3][0] = (pos.x + scale*PatchWarper::kPatchRightSize) / maxWidth;
		data.texCoordinates[3][1] = (pos.y + scale*PatchWarper::kPatchRightSize) / maxHeight;

		if(isMatched)
			mMatchedFeaturesToDraw.push_back(data);
		else
			mFeaturesToDraw.push_back(data);
	}

    //Key frames
    for(auto &framePtr : mRegion->getKeyFrames())
    {
    	//mFrustumsToDraw.push_back(prepareFrameFrustum((*it)->getPose(), (*it)->getCameraModel(), false));
    	auto &tex = getFrameTexture(*framePtr);
    	mFrustumsToDraw.push_back(prepareFrameFrustum(framePtr->getPose(), framePtr->getCameraModel(), framePtr->mOriginalRegionID, true, tex.getTarget(), tex.getId()));
    }

    //Draw currently tracked frame
	if(&mTracker->getActiveRegion() == mRegion)
	{
		mFrustumsToDraw.push_back(prepareFrameFrustum(mTracker->getCurrentPose(), mTracker->getCamera(), -1, true, mCurrentImageTextureTarget, mCurrentImageTextureId));
	}
}

bool MapWindow::isFeatureMatched(const SlamFeature &feature)
{
	auto match = mSlam->getTracker().getMatch(&feature);
	if(match)
		return true;
	return false;
}

void MapWindow::resize()
{
	auto &refCamera = mRegion->getKeyFrames()[0]->getCameraModel();
	cv::Size screenSize = UserInterfaceInfo::Instance().getScreenSize();

	mViewerCamera.init(refCamera.getFx(), refCamera.getFy(), (float)screenSize.width / 2, (float)screenSize.height / 2, screenSize.width, screenSize.height);
}

void MapWindow::ensureValidRegion()
{
	bool isValid=false;
	mMap = &mSlam->getMap();
	for(auto &region : mMap->getRegions())
	{
		if(mRegion == region.get())
		{
			isValid = true;
			break;
		}
	}

	if(!isValid)
		mRegion = mMap->getRegions().front().get();
}

void MapWindow::selectNextRegion()
{
	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	assert(!mMap->getRegions().empty());

	auto it = mMap->getRegions().begin();
	auto end = mMap->getRegions().end();
	for(; it!=end; ++it)
	{
		if(it->get() == mRegion)
			break;
	}

	if(it!=end)
		++it;

	if(it==end)
		mRegion = mMap->getRegions().front().get();
	else
		mRegion = it->get();

	updateState();
}

void MapWindow::startCube()
{
	mActiveDragType = EDragType::DraggingCube;
}

void MapWindow::updateCube(const cv::Point2f &origin, const cv::Point2f &end)
{
	std::vector<cv::Vec3f> points;

	for(auto &feature : mFeaturesToDraw)
	{
		cv::Point3f center(feature.center[0], feature.center[1], feature.center[2]);
		cv::Point2f screenPos = mViewerCamera.projectFromWorld(mViewerPose.apply(center));

		if(screenPos.x >= origin.x && screenPos.y >= origin.y && screenPos.x <= end.x && screenPos.y <= end.y)
			points.push_back(center);
	}

	if(points.size() < 3)
	{
		mApp->disableARCube();
		return;
	}

	cv::Mat pointsMat(points.size(), 3, CV_32F, points.data());
	cv::Matx33f covar;
	cv::Matx13f mean;
	cv::calcCovarMatrix(pointsMat, covar, mean, cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE, CV_32F);
	//DTSLAM_LOG << covar << "\n";

	cv::Point3f pointsMean(mean(0,0), mean(0,1), mean(0,2));

	//Decompose
	cv::SVD svd(covar, cv::SVD::MODIFY_A);
	//DTSLAM_LOG << svd.u << svd.w << svd.vt << "\n";
	cv::Point3f axis1(svd.u.at<float>(0,0),svd.u.at<float>(1,0),svd.u.at<float>(2,0));
	cv::Point3f axis2(svd.u.at<float>(0,1),svd.u.at<float>(1,1),svd.u.at<float>(2,1));
	cv::Point3f axis3(svd.u.at<float>(0,2),svd.u.at<float>(1,2),svd.u.at<float>(2,2));

	float maxEigenValue = svd.w.at<float>(0,0);
	float cubeSize = 3*sqrtf(maxEigenValue);

	axis1 *= cubeSize/2;
	axis2 *= cubeSize/2;
	axis3 *= cubeSize/2;

	//Decide normal direction
	cv::Point3f cubeCenterA = pointsMean + axis3;
	cv::Point3f cubeCenterB = pointsMean - axis3;

	cv::Point3f cubeProjA = mViewerPose.apply(cubeCenterA);
	cv::Point3f cubeProjB = mViewerPose.apply(cubeCenterB);

	cv::Vec3f cubeCenter = (cubeProjA.z < cubeProjB.z) ? cubeCenterA : cubeCenterB;

	//Update
	mApp->setARCube(cubeCenter, axis1, axis2, axis3);
}

void MapWindow::touchDown(int id, int x, int y)
{
	if(mActiveDragType == EDragType::NoDragging)
	{
		//No dragging active, start a new one
		switch(id)
		{
		case kMouseLeftButton:
		case kMouseRightButton:
			mActiveDragType = (id==kMouseLeftButton) ? EDragType::DraggingRotation : EDragType::DraggignTranslation;
			mDragOrigin = cv::Point2f((float)x, (float)y);
			mDragStartingPose = mViewerPose;
			break;

		case kMouseScrollDown: zoomIn(); break;
		case kMouseScrollUp: zoomOut(); break;
		}
	}
	else if(mActiveDragType == EDragType::DraggingCube)
	{
		//Start the cube
		mDragOrigin = cv::Point2f((float)x, (float)y);
	}
}

void MapWindow::zoomIn()
{
	zoom(+5.0f);
}

void MapWindow::zoomOut()
{
	zoom(-5.0f);
}

void MapWindow::zoom(float ammount)
{
	//Zoom
	const cv::Vec3f oldT = mViewerPose.getTranslationRef();
	mViewerPose.setTranslation(
		cv::Vec3f(oldT[0],
		oldT[1],
		oldT[2] + ammount / kTranslateScale));
	{
		shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());
		updateState();
	}
}

void MapWindow::touchMove(int x, int y)
{
	mDragEnd = cv::Point2f((float)x, (float)y);
	if(mActiveDragType == EDragType::DraggingRotation)
	{
		//Rotation
		cv::Point3f originXn = mViewerCamera.unprojectToWorld(mDragOrigin);
		cv::Point3f endXn = mViewerCamera.unprojectToWorld(mDragEnd);

		float angleX = 2 * asinf(originXn.x-endXn.x);
		float angleY = 2 * asinf(endXn.y - originXn.y);

		cv::Matx33f rotX = cvutils::RotationX(angleX);
		cv::Matx33f rotY = cvutils::RotationY(angleY);

		cv::Matx33f rotR = rotX * rotY;
		mViewerPose.set(rotR * mDragStartingPose.getRotationRef(), rotR * mDragStartingPose.getTranslationRef());
        {
        	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());
        	updateState();
        }
	}
	else if(mActiveDragType == EDragType::DraggignTranslation)
	{
		//Translation
        const cv::Vec3f oldT = mDragStartingPose.getTranslationRef();
        mViewerPose.setTranslation(
                cv::Vec3f(oldT[0] + (mDragEnd.x - mDragOrigin.x) / kTranslateScale,
                          oldT[1] + (mDragEnd.y - mDragOrigin.y) / kTranslateScale,
                          oldT[2]));

        {
        	shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());
        	updateState();
        }
	}
	else if(mActiveDragType == EDragType::DraggingCube)
	{
		//Update final cube
		updateCube(mDragOrigin, mDragEnd);
	}
}

void MapWindow::touchUp(int id, int x, int y)
{
	mActiveDragType = EDragType::NoDragging;
}

void MapWindow::draw()
{
	cv::Size screenSize = UserInterfaceInfo::Instance().getScreenSize();

	cv::Matx34f KRt3 = mViewerCamera.getK() * mViewerPose.getRt();
	cv::Matx44f KRt;
	for(int i=0; i<3*4; ++i)
		KRt.val[i] = KRt3.val[i];
	KRt(3,0) = KRt(3,1) = KRt(3,2) = 0;
	KRt(3,3) = 1;

	cv::Matx44f mvp = ViewportTiler::GetImageSpaceMvp(screenSize, mViewerCamera.getImageSize()) * KRt;
	mShaders->getColor().setMVPMatrix(mvp);
	mShaders->getTexture().setMVPMatrix(mvp);

	glPointSize(mPointSize);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

	//Draw pose log
	mShaders->getColor().drawVertices(GL_LINE_STRIP, mPoseLog.data(), mPoseLog.size(), StaticColors::Red());

    //Draw features
    for(auto &data : mFeaturesToDraw)
    {
    	drawFeature(data);
    }

    glDisable(GL_DEPTH_TEST);
    for(auto &data : mMatchedFeaturesToDraw)
    {
    	drawFeature(data);
    }
    glEnable(GL_DEPTH_TEST);

    //Draw frustums
    for(auto &frameData : mFrustumsToDraw)
    {
    	drawFrameFrustum(frameData);
    }

    //Draw cube
    if(mApp->isARCubeValid())
    {
    	std::vector<unsigned int> triangleIndices;
    	std::vector<cv::Vec4f> vertices;
    	std::vector<cv::Vec4f> colors;
    	std::vector<cv::Vec3f> normals;
        mApp->generateARCubeVertices(triangleIndices, vertices, colors, normals);

    	mShaders->getColor().drawVertices(GL_TRIANGLES, triangleIndices.data(), triangleIndices.size(), vertices.data(), colors.data());
    }

    glDisable(GL_DEPTH_TEST);

    //Draw current frame as reference
    const int drawHeight = 100;
    float imgAspect = (float)mImageSize.width / mImageSize.height;
    cv::Size drawSize((int)(imgAspect*drawHeight),drawHeight);
    mShaders->getTexture().setMVPMatrix(ViewportTiler::GetImageSpaceMvp(screenSize, screenSize));
    mShaders->getTexture().renderTexture(mCurrentImageTextureTarget,mCurrentImageTextureId,drawSize,cv::Point2i(screenSize.width-drawSize.width, screenSize.height-drawSize.height));
}

void MapWindow::drawFeature(const DrawFeatureData &data)
{
	cv::Vec4f solidColor = data.solidColor;

	//Special color when dragging for cube
	if(mActiveDragType == EDragType::DraggingCube)
	{
		cv::Point3f center(data.center[0], data.center[1], data.center[2]);
		cv::Point2f screenPos = mViewerCamera.projectFromWorld(mViewerPose.apply(center));

		if(screenPos.x >= mDragOrigin.x && screenPos.y >= mDragOrigin.y && screenPos.x <= mDragEnd.x && screenPos.y <= mDragEnd.y)
			solidColor = StaticColors::White(1);
	}

	if(mDrawFeaturePatches)
	{
		glDepthMask(false);
		//mShaders->getColor().drawVertices(GL_TRIANGLE_STRIP, vertices, 4, StaticColors::White(0.55f));

		if(data.useTex)
			mShaders->getTexture().renderTexture(GL_TRIANGLE_STRIP, data.texTarget, data.texId, data.vertices, data.texCoordinates, 4, 0.55f);
		else
			mShaders->getColor().drawVertices(GL_TRIANGLE_STRIP, data.vertices, 4, solidColor);

	    unsigned int indices[] = { 0, 1, 3, 2};
	    cv::Vec4f colors[] = {data.solidColor,data.solidColor,data.solidColor,solidColor};
	    mShaders->getColor().drawVertices(GL_LINE_LOOP, indices, 4, data.vertices, colors);

		glDepthMask(true);
	}
	else
	{
		mShaders->getColor().drawVertices(GL_POINTS, &data.center, 1, solidColor);
	}
}

const TextureHelper &MapWindow::getFrameTexture(const SlamKeyFrame &frame)
{
	auto it = mFrameTextures.find(&frame);
	if(it == mFrameTextures.end())
	{
		auto add = mFrameTextures.insert(std::make_pair(&frame, std::unique_ptr<TextureHelper>(new TextureHelper())));
		TextureHelper &texture = *add.first->second;
		texture.create(GL_RGB, frame.getColorImage().size());
		texture.update(frame.getColorImage());
		return texture;
	}
	else
		return *it->second;
}

MapWindow::DrawFrustumData MapWindow::prepareFrameFrustum(const Pose3D &pose, const CameraModel &camera, int regionId, bool useTex, unsigned int texTarget, unsigned int texID)
{
	DrawFrustumData data;
    const float kFrustumDepth = 0.3f*mMapDrawScale;

    data.tl = camera.unprojectToWorld(cv::Point2f(0, 0)) * kFrustumDepth;
	data.tr = camera.unprojectToWorld(cv::Point2f((float)camera.getImageSize().width, 0)) * kFrustumDepth;
	data.bl = camera.unprojectToWorld(cv::Point2f(0, (float)camera.getImageSize().height)) * kFrustumDepth;
	data.br = camera.unprojectToWorld(cv::Point2f((float)camera.getImageSize().width, (float)camera.getImageSize().height)) * kFrustumDepth;

    data.center = -pose.getRotation().t() * pose.getTranslation();
    data.tl = pose.applyInv(data.tl);
    data.tr = pose.applyInv(data.tr);
    data.bl = pose.applyInv(data.bl);
    data.br = pose.applyInv(data.br);

	if (regionId < 0)
		data.color = StaticColors::Red();
	else
	    data.color = mRegionColors[regionId % mRegionColors.size()];

    data.useTex = useTex;
    data.texTarget = texTarget;
    data.texId = texID;

    return data;
}

void MapWindow::drawFrameFrustum(const DrawFrustumData &data)
{
	std::array<cv::Vec4f,5> colors;
	colors.fill(data.color);

    cv::Vec4f vertices[5] =
    { cvutils::PointToHomogenous(data.tl), cvutils::PointToHomogenous(data.tr),
    		cvutils::PointToHomogenous(data.bl), cvutils::PointToHomogenous(data.br),
    		cvutils::PointToHomogenous(data.center)};
    unsigned int indices[8] =
    { 0, 1, 4, 0, 2, 4, 3, 1 };

    mShaders->getColor().drawVertices(GL_LINE_STRIP, indices, 8, vertices, colors.data());

    cv::Vec4f v2[] =
    { vertices[1], vertices[0], vertices[3], vertices[2] };

    glDepthMask(false);
    if (data.useTex)
    {
        cv::Vec2f textureCoords[] =
        { cv::Vec2f(1, 0), cv::Vec2f(0, 0), cv::Vec2f(1, 1), cv::Vec2f(0, 1) };

        mShaders->getTexture().renderTexture(GL_TRIANGLE_STRIP, data.texTarget, data.texId, v2, textureCoords, 4, 0.55f);
    }
    else
    {
        //Use this to show only solid gray
        colors.fill(StaticColors::Gray(0.55f,0.75f));
        mShaders->getColor().drawVertices(GL_TRIANGLE_STRIP, v2, colors.data(), 4);
    }
    glDepthMask(true);

}

void MapWindow::performBA()
{
	mPerformBAFuture = std::async(std::launch::async, PerformBATask, this);
}

void MapWindow::PerformBATask(MapWindow *window)
{
	window->performBATask();
}

void MapWindow::performBATask()
{
	Profiler::Instance().setCurrentThreadName("mapBundleAdjuster");

	BundleAdjuster ba;
	ba.setUseLocks(true);
	ba.setOutlierThreshold((float)FLAGS_TrackerOutlierPixelThreshold);

	{ //Read-lock to prepare BA
		shared_lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

		ensureValidRegion();

		ba.setRegion(mMap, mRegion);
		for(auto &frame : mRegion->getKeyFrames())
			ba.addFrameToAdjust(*frame);
	}

	//BA will lock on its own
	DTSLAM_LOG << "BA requested manually.";
	ba.bundleAdjust();
	DTSLAM_LOG << "Manual BA finished.";
}

void MapWindow::forceNewKeyFrame()
{
	mSlam->getMapExpander().addKeyFrame();
}

} /* namespace dtslam */
