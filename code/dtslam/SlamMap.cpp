/*
 * SlamMap.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "SlamMap.h"
#include <map>
#include <cassert>
#include <opencv2/calib3d.hpp>
#include "SlamKeyFrame.h"
#include "Pose3D.h"

#include "cvutils.h"
#include "Profiler.h"
#include "EssentialUtils.h"

#include "flags.h"

namespace dtslam {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SlamMap
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

	SlamRegion *SlamMap::createRegion()
{
	SlamRegion *region = new SlamRegion();
	region->setId(mNextRegionId++);
	mRegions.push_back(std::unique_ptr<SlamRegion>(region));
	return region;
}

int SlamMap::getTotalFrameCount() const
{
	int count=0;
	for(auto &region : mRegions)
		count += region->getKeyFrames().size();
	return count;
}

int SlamMap::getTotalFeature3DCount() const
{
	int count=0;
	for(auto &region : mRegions)
		count += region->getFeatures3D().size();
	return count;
}

int SlamMap::getTotalFeature2DCount() const
{
	int count=0;
	for(auto &region : mRegions)
		count += region->getFeatures2D().size();
	return count;
}

void SlamMap::clear()
{
	mRegions.clear();
}

void SlamMap::mergeRegions(SlamRegion &regionA, SlamRegion &regionB, const cv::Matx33f &Rrel, float scale, const cv::Vec3f &trel)
{
	for (auto &featurePtr : regionB.mFeatures3D)
	{
		auto &feature = *featurePtr;
		feature.setPosition(scale*Rrel*feature.getPosition() + cv::Point3f(trel));

		featurePtr->setRegion(&regionA);
		regionA.mFeatures3D.emplace_back(std::move(featurePtr));
	}
	for (auto &featurePtr : regionB.mFeatures2D)
	{
		auto &feature = *featurePtr;
		//feature.setPosition(scale*Rrel*feature.getPosition() + cv::Point3f(trel));

		featurePtr->setRegion(&regionA);
		regionA.mFeatures2D.emplace_back(std::move(featurePtr));
	}
	for (auto &framePtr : regionB.mKeyFrames)
	{
		auto &frame = *framePtr;
		Pose3D *pose = &frame.getPose();
		FullPose3D *fullpose = dynamic_cast<FullPose3D*>(pose);
		if (fullpose)
		{
			cv::Matx33f newR = fullpose->getRotationRef() * Rrel.t();
			cv::Vec3f newT = -newR * trel + scale*fullpose->getTranslationRef();
			fullpose->set(newR, newT);
		}

		framePtr->setRegion(&regionA);
		regionA.mKeyFrames.emplace_back(std::move(framePtr));
	}

	//Remove regionB
	for (auto it = mRegions.begin(), end = mRegions.end(); it != end; ++it)
	{
		if (it->get() == &regionB)
		{
			mRegions.erase(it);
			break;
		}
	}
}

void SlamMap::serialize(Serializer &s, cv::FileStorage &fs) const
{
	fs << "nextRegionId" << mNextRegionId;

	fs << "regions" << "[";
	for (auto &p : mRegions)
		fs << s.addObject(p.get());
	fs << "]";
}

void SlamMap::deserialize(Deserializer &s, const cv::FileNode &node)
{
	node["nextRegionId"] >> mNextRegionId;

	for (const auto &n : node["regions"])
	{
		mRegions.push_back(s.getObjectForOwner<SlamRegion>(n));
	}

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SlamRegion
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SlamRegion::addKeyFrame(std::unique_ptr<SlamKeyFrame> newKeyFrame)
{
	if(mFirstTriangulationFrame==NULL && !mKeyFrames.empty())
	{
		cv::Vec3f centerRef = mKeyFrames.front()->getPose().getCenter();
		cv::Vec3f centerNew = newKeyFrame->getPose().getCenter();
		if(cvutils::PointDistSq(centerRef, centerNew) != 0)
			mFirstTriangulationFrame = newKeyFrame.get();
	}

	//Add key frame to map
	newKeyFrame->setRegion(this);
	mKeyFrames.push_back(std::move(newKeyFrame));
}
void SlamRegion::addFeature3D(std::unique_ptr<SlamFeature> newFeature)
{
	mFeatures3D.push_back(std::move(newFeature));
}
void SlamRegion::getFeaturesInView(const Pose3D &pose, const CameraModel &camera, const int octaveCount, bool onlyNearest2DSection, const std::unordered_set<SlamFeature*> &featuresToIgnore, std::vector<std::vector<FeatureProjectionInfo>> &featuresInView)
{
	ProfileSection s("getFeaturesInView");

	const cv::Size &imageSize = camera.getImageSize();
	cv::Vec3f poseCenter = pose.getCenter();

	featuresInView.clear();
	featuresInView.resize(octaveCount);

	//Add 3D features
	for(auto it=mFeatures3D.begin(); it!=mFeatures3D.end(); it++)
	{
		SlamFeature &feature = **it;

		//Ignore?
		if (featuresToIgnore.find(&feature) != featuresToIgnore.end())
			continue;

		FeatureProjectionInfo projection = Project3DFeature(pose, poseCenter, camera, octaveCount, feature);
		if (projection.getType() != EProjectionType::Invalid)
			featuresInView[projection.getOctave()].push_back(projection);

		////Project position to see if it is within the image
		//const cv::Point3f xc = pose.apply(feature.getPosition());
		//if(xc.z <= 0)
		//	continue; //Point behind the camera

		//const cv::Point2f uv = camera.projectFromWorld(xc);

		//if(uv.x >= 0 && uv.x < imageSize.width
		//		&& uv.y >= 0 && uv.y < imageSize.height)
		//{
		//	//Project positionPlusOne to determine scale
		//	const cv::Point2f uv1 = camera.projectFromWorld(pose.apply(feature.getPositionPlusOne()));

		//	int distSq = static_cast<int>(cvutils::PointDistSq(uv,uv1)+0.5f);
		//	if(distSq < 1)
		//		continue; //Scale incompatible

		//	int octave = 0;
		//	while(distSq> 1 && octave < octaveCount)
		//	{
		//		distSq >>= 2;
		//		octave++;
		//	}
		//	if(octave >= octaveCount)
		//		continue; //Scale incompatible
		//featuresInView[octave].push_back(FeatureProjectionInfo::CreatePoint(&feature, feature.getBestMeasurementForMatching(poseCenter), octave, uv));
	}

	//2D features
	std::vector<std::pair<float, SlamKeyFrame*>> frames;
	int frameCount;
	if (!onlyNearest2DSection || (int)frames.size() < FLAGS_FramesFor2DTracking)
	{
		//Add all frames
		for (auto &framePtr : mKeyFrames)
		{
			frames.push_back(std::make_pair(0.0f, framePtr.get()));
		}
		frameCount = frames.size();
	}
	else
	{
		//Add only closest frames
		//Calculate distance
		for (auto &framePtr : mKeyFrames)
		{
			auto &frame = *framePtr;
			cv::Vec3f frameCenter = frame.getPose().getCenter();

			float distSq = cvutils::PointDistSq(frameCenter, poseCenter);
			frames.push_back(std::make_pair(distSq, &frame));
		}

		//Sort
		std::nth_element(frames.begin(), frames.begin() + FLAGS_FramesFor2DTracking, frames.end(),
			[](const std::pair<float, SlamKeyFrame*>& a, const std::pair<float, SlamKeyFrame*>& b) {
			return a.first < b.first; });

		frameCount = FLAGS_FramesFor2DTracking;
	}

	std::unordered_map<SlamFeature *, SlamFeatureMeasurement *> features2D;
	for (int i = 0; i<frameCount; ++i)
	{
		for(auto &mPtr : frames[i].second->getMeasurements())
		{
			auto &m = *mPtr;

			//Is 2D?
			if(m.getFeature().is3D())
				continue;

			//Only exact matches can be used as 2D sources
			if(m.getPositions().size()>1)
				continue;

			//Ignore?
			if (featuresToIgnore.find(&m.getFeature()) != featuresToIgnore.end())
				continue;

			features2D.insert(std::make_pair(&m.getFeature(), &m));
		}
	}

	for(auto &featurePair : features2D)
	{
		auto &m = *featurePair.second;
		FeatureProjectionInfo projection = Project2DFeature(pose, poseCenter, camera, m);
		if (projection.getType() != EProjectionType::Invalid)
			featuresInView[projection.getOctave()].push_back(projection);
	}
}

FeatureProjectionInfo SlamRegion::Project3DFeature(const Pose3D &pose, const cv::Vec3f &poseCenter, const CameraModel &camera, int octaveCount, const SlamFeature &feature)
{
	//Project position to see if it is within the image
	const cv::Point3f xc = pose.apply(feature.getPosition());
	const cv::Point2f uv = camera.projectFromWorld(xc);

	if(!camera.isPointInside(xc,uv))
		return FeatureProjectionInfo();

	//Project positionPlusOne to determine scale
	const cv::Point2f uv1 = camera.projectFromWorld(pose.apply(feature.getPositionPlusOne()));

	int distSq = static_cast<int>(cvutils::PointDistSq(uv,uv1)+0.5f);
	if(distSq < 1)
		return FeatureProjectionInfo(); //Scale incompatible

	int octave = 0;
	while(distSq> 1 && octave < octaveCount)
	{
		distSq >>= 2;
		octave++;
	}
	if(octave >= octaveCount)
		return FeatureProjectionInfo(); //Scale incompatible

	return FeatureProjectionInfo::CreatePoint(const_cast<SlamFeature*>(&feature), feature.getBestMeasurementForMatching(poseCenter), octave, uv);
}

FeatureProjectionInfo SlamRegion::Project2DFeature(const Pose3D &pose, const cv::Vec3f &poseCenter, const CameraModel &camera, SlamFeatureMeasurement &m)
{
	int octave = m.getOctave();

	//Get fundamental matrix
	const Pose3D &refPose = m.getKeyFrame().getPose();
	//cv::Vec3f refCenter = refPose.getCenter();

	FullPose3D relativePose = FullPose3D::MakeRelativePose(refPose, pose);

	//Project infinite position
	const cv::Point3f infiniteXn = relativePose.getRotationRef() * m.getUniquePositionXn();
	const cv::Point2f infiniteUv = camera.projectFromWorld(infiniteXn);

	//Check projection is in image
	if(!camera.isPointInside(infiniteXn, infiniteUv))
		return FeatureProjectionInfo();

	//Determine motion model
	if(cvutils::NormSq(relativePose.getTranslationRef()) == 0)
	{
		//Pure rotation
		return FeatureProjectionInfo::CreatePoint(&m.getFeature(), &m, octave, infiniteUv);
	}
	else
	{
		//Get min depth
		const cv::Point3f refMinDepthX = m.getUniquePositionXn() * FLAGS_MinDepth;
		const cv::Point3f minDepthX = m.getKeyFrame().getPose().applyInv(refMinDepthX);
		const cv::Point3f imgMinDepthX = pose.apply(minDepthX);
		const cv::Point3f imgMinDepthXn = cvutils::PointToUnitNorm(imgMinDepthX);


		//TODO: no need to calculate epiPlaneNormal from the epipolar matrix, calculate instead from the triangle: mindepth, infinite, center
		//Full essential model
		cv::Matx33f E = EssentialUtils::EssentialFromPose(relativePose);

		//Determine epipolar line
		cv::Vec3f epiPlaneNormal = cvutils::PointToUnitNorm(E*m.getUniquePositionXn());

		return FeatureProjectionInfo::CreateEpipolar(&m.getFeature(), &m, octave, epiPlaneNormal, imgMinDepthXn, infiniteXn);
	}
}

EpipolarProjection SlamRegion::CreateEpipolarProjection(const Pose3D &refPose, const cv::Point3f refXn, const Pose3D &imgPose)
{
	EpipolarProjection projection;

	FullPose3D relativePose = FullPose3D::MakeRelativePose(refPose, imgPose);

	//Project infinite position
	projection.infiniteXn = relativePose.getRotationRef() * refXn;

	//Get min depth
	const cv::Point3f refMinDepthX = refXn * FLAGS_MinDepth;
	const cv::Point3f minDepthX = refPose.applyInv(refMinDepthX);
	const cv::Point3f imgMinDepthX = imgPose.apply(minDepthX);
	projection.minDepthXn = cvutils::PointToUnitNorm(imgMinDepthX);

	//Determine epipolar line
	projection.epiPlaneNormal = cvutils::PointToUnitNorm(projection.minDepthXn.cross(projection.infiniteXn));

	return projection;
}

SlamFeature *SlamRegion::createFeature2D(SlamKeyFrame &keyFrame, const cv::Point2f &position, const cv::Point3f &positionXn, int octave)
{
	//FullPose3D poseRelative = FullPose3D::MakeRelativePose(mRootFrame->getPose(), keyFrame.getPose());
	//assert(cvutils::NormSq(poseRelative.getTranslationRef()) < 0.1f*0.1f);
	cv::Matx33f Rt = keyFrame.getPose().getRotation().t();

	const int scale=1<<octave;

	auto &camera = keyFrame.getCameraModel();

	std::unique_ptr<SlamFeature> feature(new SlamFeature());
	feature->mOriginalRegionId = mId; //Remember the first for coloring
	feature->setRegion(this);
	feature->mIs3D = false;
	feature->setStatus(SlamFeatureStatus::NotTriangulated);
	//feature->mPosition = SlamMap::Unproject2d(position, keyFrame.getCameraModel(), Rt); //Position should not be set and cannot be queried until triangulated
	feature->mPosition = Rt*camera.unprojectToWorld(position); //Position should not be set and cannot be queried until triangulated
	feature->mNormal = feature->mPosition;

	const cv::Point3f positionPlusOne = Rt*camera.unprojectToWorld(cv::Point2f(position.x + scale, position.y));
	feature->mPlusOneOffset = positionPlusOne-feature->mPosition;

	feature->mMeasurements.push_back(std::unique_ptr<SlamFeatureMeasurement>(new SlamFeatureMeasurement(feature.get(), &keyFrame, std::vector<cv::Point2f>(1,position), std::vector<cv::Point3f>(1,positionXn), octave)));

	keyFrame.getMeasurements().push_back(feature->getMeasurements()[0].get());

	SlamFeature *ptr = feature.get();
	mFeatures2D.push_back(std::move(feature));
	return ptr;
}


void SlamRegion::convertTo3D(SlamFeature &feature, SlamFeatureMeasurement &m1, SlamFeatureMeasurement &m2)
{
	assert(!feature.is3D());
	assert(&m1.getFeature()==&feature && &m2.getFeature()==&feature);
	assert(m1.getPositions().size()==1 && m2.getPositions().size()==1);

	//Triangulate (in m1 frame coordinates)
	const Pose3D &refPose = m1.getKeyFrame().getPose();
	FullPose3D poseAB = FullPose3D::MakeRelativePose(refPose, m2.getKeyFrame().getPose());

	cv::Matx34f P1 = cv::Matx34f::eye();
	cv::Matx34f P2 = poseAB.getRt();

	std::vector<cv::Point2f> refXn2(1,cvutils::NormalizePoint(m1.getUniquePositionXn()));
	std::vector<cv::Point2f> imgXn2(1,cvutils::NormalizePoint(m2.getUniquePositionXn()));
	cv::Mat1f p4mat;
	cv::triangulatePoints(P1, P2, refXn2, imgXn2, p4mat);
	assert(fabs(p4mat(3,0)) > 0.0001f);

	const cv::Point3f positionRelative = cv::Point3f(p4mat(0,0)/p4mat(3,0), p4mat(1,0)/p4mat(3,0), p4mat(2,0)/p4mat(3,0));
	const float depth = sqrtf(cvutils::NormSq(positionRelative));

	//Translate to world coordinates
	feature.mPosition = refPose.applyInv(positionRelative);
	feature.mIs3D = true;
	feature.setStatus(SlamFeatureStatus::TwoViewTriangulation);

	//Calculate position plus one
	const int scale = 1<<m1.getOctave();
	const cv::Point2f uv1 = cv::Point2f(m1.getUniquePosition().x+scale,m1.getUniquePosition().y);
	const cv::Point3f plusOne = depth * m1.getKeyFrame().getCameraModel().unprojectToWorld(uv1);
	const cv::Point3f plusOneWorld = refPose.applyInv(plusOne);

	const cv::Point3f center = depth * m1.getKeyFrame().getCameraModel().unprojectToWorld(m1.getUniquePosition());
	const cv::Point3f centerWorld = refPose.applyInv(center);
	feature.mPlusOneOffset = plusOneWorld - centerWorld;

	//Move from 2D section to general vector
	auto itFeature = mFeatures2D.begin();
	for(;itFeature!=mFeatures2D.end(); ++itFeature)
		if(itFeature->get() == &feature)
			break;
	assert(itFeature != mFeatures2D.end());

	mFeatures3D.push_back(std::move(*itFeature));
	mFeatures2D.erase(itFeature);
}

void SlamRegion::moveToGarbage(SlamFeature &feature)
{
	auto *featureList = &mFeatures2D;
	if(feature.is3D())
		featureList = &mFeatures3D;
	
	//Mark as invalid
	feature.setStatus(SlamFeatureStatus::Invalid);

	//Remove from list
	auto it = featureList->begin(), end = featureList->end();
	for (; it != end; ++it)
	{
		if (it->get() == &feature)
			break; //Found!
	}

	if (it != end)
	{
		mGarbageFeatures.push_back(std::move(*it));
		featureList->erase(it);
	}
	else
	{
		DTSLAM_LOG << "Ahhh!!! Attempt to move a feature to the garbage that is not in the map (measurement count: " << feature.getMeasurements().size() << ").\n";
#if _MSC_VER
		__debugbreak();
#else
                assert(false);
#endif
	}

	//Remove from frames
	for (auto &mPtr : feature.getMeasurements())
	{
		auto &frame = mPtr->getKeyFrame();
		frame.removeMeasurement(mPtr.get());
	}

	//Features are in the garbage, don't clear, breaks other code
	//TODO: this shouldn't break other code
	//feature.mMeasurements.clear();
}

void SlamRegion::serialize(Serializer &s, cv::FileStorage &fs) const
{
	fs << "id" << mId;

	fs << "keyframes" << "[";
	for (auto &p: mKeyFrames)
		fs << s.addObject(p.get());
	fs << "]";

	fs << "features2D" << "[";
	for (auto &p : mFeatures2D)
		fs << s.addObject(p.get());
	fs << "]";
	fs << "features3D" << "[";
	for (auto &p : mFeatures3D)
		fs << s.addObject(p.get());
	fs << "]";

	fs << "firstTriangulation" << s.addObject(mFirstTriangulationFrame);
}

void SlamRegion::deserialize(Deserializer &s, const cv::FileNode &node)
{
	node["id"] >> mId;

	for (const auto &n : node["keyframes"])
	{
		mKeyFrames.push_back(s.getObjectForOwner<SlamKeyFrame>(n));
	}

	for (const auto &n : node["features2D"])
	{
		mFeatures2D.push_back(s.getObjectForOwner<SlamFeature>(n));
	}

	for (const auto &n : node["features3D"])
	{
		mFeatures3D.push_back(s.getObjectForOwner<SlamFeature>(n));
	}

	mFirstTriangulationFrame = s.getObject<SlamKeyFrame>(node["firstTriangulation"]);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SlamFeature
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SlamFeature::setStatus(int inlierMeasurementCount)
{
	int mcount = mMeasurements.size();
	if (mcount == 2)
	{
		mStatus = SlamFeatureStatus::TwoViewTriangulation;
	}
	else if (mcount == 3)
	{
		if (inlierMeasurementCount > 2)
			mStatus = SlamFeatureStatus::ThreeViewAgreement;
		else
			mStatus = SlamFeatureStatus::ThreeViewDisagreement;
	}
	else
	{
		if (inlierMeasurementCount == mcount)
			mStatus = SlamFeatureStatus::MultiViewAgreement;
		else
			mStatus = SlamFeatureStatus::MultiViewDisagreement;
	}
}

int SlamFeature::getOctaveFor2DFeature() const
{
	return mMeasurements[0]->getOctave();
}

float SlamFeature::GetTriangulationAngle(const SlamFeatureMeasurement &m1, const SlamFeatureMeasurement &m2)
{
	const Pose3D &poseA = m1.getKeyFrame().getPose();
	const cv::Point3f infiniteXnWorld = poseA.getRotation().t() * m1.getUniquePositionXn();

	const Pose3D &poseB = m2.getKeyFrame().getPose();
	const cv::Point3f infiniteXn = poseB.getRotation() * infiniteXnWorld;
	const float dotValue = std::min(infiniteXn.dot(m2.getUniquePositionXn()), 1.0f);
	return acosf(dotValue);
}

float SlamFeature::getMinTriangulationAngle(const SlamFeatureMeasurement &m1) const
{
	int mcount = mMeasurements.size();
	if (!mcount)
		return -1;

	const Pose3D &poseA = m1.getKeyFrame().getPose();
	const cv::Point3f infiniteXnWorld = poseA.getRotation().t() * m1.getUniquePositionXn();

	float angle = std::numeric_limits<float>::max();

	for(int j=0; j<mcount; ++j)
	{
		SlamFeatureMeasurement &mb = *mMeasurements[j];
		if(mb.getPositions().size() > 1)
			continue;

		const Pose3D &poseB = mb.getKeyFrame().getPose();

		const cv::Point3f infiniteXn = poseB.getRotation() * infiniteXnWorld;
		const float dotValue = std::min(infiniteXn.dot(mb.getUniquePositionXn()), 1.0f);
		const float angle_ij = acosf(dotValue);

		if(angle_ij < angle)
		{
			angle = angle_ij;
		}
	}

	return angle;
}

void SlamFeature::getMeasurementsForTriangulation(SlamFeatureMeasurement *&m1, SlamFeatureMeasurement *&m2, float &angle) const
{
	int mcount = mMeasurements.size();

	angle = 0;
	m1 = NULL;
	m2 = NULL;

	if(mcount < 2)
	{
		return;
	}

	for(int i=0; i<mcount; ++i)
	{
		SlamFeatureMeasurement &ma = *mMeasurements[i];
		if(ma.getPositions().size() > 1)
			continue;

		const Pose3D &poseA = ma.getKeyFrame().getPose();
		const cv::Point3f infiniteXnWorld = poseA.getRotation().t() * ma.getUniquePositionXn();

		for(int j=i+1; j<mcount; ++j)
		{
			SlamFeatureMeasurement &mb = *mMeasurements[j];
			if(mb.getPositions().size() > 1)
				continue;

			const Pose3D &poseB = mb.getKeyFrame().getPose();

			const cv::Point3f infiniteXn = poseB.getRotation() * infiniteXnWorld;
			const float angle_ij = acosf(infiniteXn.dot(mb.getUniquePositionXn()));

			if(angle_ij > angle)
			{
				angle = angle_ij;
				m1 = &ma;
				m2 = &mb;
			}
		}
	}
}

void SlamFeature::getMeasurementsForTriangulation(const SlamFeatureMeasurement &m1, SlamFeatureMeasurement *&m2, float &angle) const
{
	int mcount = mMeasurements.size();

	angle = -1;
	m2 = NULL;

	if(mcount < 1 || m1.getPositions().size() > 1)
	{
		return;
	}

	const Pose3D &poseA = m1.getKeyFrame().getPose();
	const cv::Point3f infiniteXnWorld = poseA.getRotation().t() * m1.getUniquePositionXn();

	for(int j=0; j<mcount; ++j)
	{
		SlamFeatureMeasurement &mb = *mMeasurements[j];
		if(mb.getPositions().size() > 1)
			continue;

		const Pose3D &poseB = mb.getKeyFrame().getPose();

		const cv::Point3f infiniteXn = poseB.getRotation() * infiniteXnWorld;
		const float angle_ij = acosf(infiniteXn.dot(mb.getUniquePositionXn()));

		if(angle_ij > angle)
		{
			angle = angle_ij;
			m2 = &mb;
		}
	}
}

void SlamFeature::serialize(Serializer &s, cv::FileStorage &fs) const
{
	fs << "region" << s.addObject(mRegion);
	fs << "is3D" << mIs3D;
	fs << "status" << (int)mStatus;

	fs << "position" << mPosition;
	fs << "normal" << mNormal; 
	fs << "plusOneOffset" << mPlusOneOffset;

	fs << "measurements" << "[";
	for (auto &m : mMeasurements)
		fs << s.addObject(m.get());
	fs << "]";
}

void SlamFeature::deserialize(Deserializer &s, const cv::FileNode &node)
{
	node["position"] >> mPosition;
	node["normal"] >> mNormal;
	node["plusOneOffset"] >> mPlusOneOffset;

	int statusInt;
	node["status"] >> statusInt;
	mStatus = (SlamFeatureStatus)statusInt;

	node["is3D"] >> mIs3D;
	mRegion = s.getObject<SlamRegion>(node["region"]);

	for (const auto &n : node["measurements"])
	{
		mMeasurements.push_back(s.getObjectForOwner<SlamFeatureMeasurement>(n));
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SlamFeatureMeasurement
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SlamFeatureMeasurement::serialize(Serializer &s, cv::FileStorage &fs) const
{
	fs << "feature" << s.addObject(mFeature);
	fs << "keyframe" << s.addObject(mKeyFrame);

	fs << "positions" << mPositions;
	fs << "octave" << mOctave;
}

void SlamFeatureMeasurement::deserialize(Deserializer &s, const cv::FileNode &node)
{
	node["octave"] >> mOctave;
	node["positions"] >> mPositions;

	mFeature = s.getObject<SlamFeature>(node["feature"]);
	mKeyFrame = s.getObject<SlamKeyFrame>(node["keyframe"]);

	int pcount = mPositions.size();
	mPositionXns.resize(pcount);
	for (int i = 0; i != pcount; ++i)
		mPositionXns[i] = getCamera().unprojectToWorld(mPositions[i]);
}

} /* namespace dtslam */
