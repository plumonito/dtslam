/*
 * FrameLinkWindow.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "FrameLinkWindow.h"
#include "dtslam/SlamSystem.h"
#include "../shaders/DTSlamShaders.h"

namespace dtslam
{

FrameLinkWindow::~FrameLinkWindow()
{

}

bool FrameLinkWindow::init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize)
{
	if(!TwoFrameWindow::init(app,slam,imageSize))
		return false;

	mKeyBindings.addBinding(false, 'm', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&FrameLinkWindow::toggleShowWarp), "Show warp.");
	mKeyBindings.addBinding(false, 'l', static_cast<KeyBindingHandler<BaseWindow>::SimpleBindingFunc>(&FrameLinkWindow::runRegionLink), "Run region link algorithm.");

	mLinker.init(&slam->getTracker().getCamera());

	return true;
}

void FrameLinkWindow::showHelp() const
{
	BaseWindow::showHelp();
	DTSLAM_LOG << "Uses the frame linker to align the two frames. This is experimental.\n";
}

void FrameLinkWindow::toggleShowWarp()
{
	mShowWarp = !mShowWarp;
}

void FrameLinkWindow::updateState(const SlamKeyFrame &frameA, const SlamKeyFrame &frameB)
{
	mOctaveCount = frameA.getPyramid().getOctaveCount();

	mLinkSuccesfull = (bool)mLinker.findLink(frameA, frameB);

	mHomography(0,0) = mLinker.getSimilarity()(0,0);
	mHomography(0,1) = mLinker.getSimilarity()(0,1);
	mHomography(0,2) = mLinker.getSimilarity()(0,2);

	mHomography(1,0) = mLinker.getSimilarity()(1,0);
	mHomography(1,1) = mLinker.getSimilarity()(1,1);
	mHomography(1,2) = mLinker.getSimilarity()(1,2);

	mHomography(2,0) = 0;
	mHomography(2,1) = 0;
	mHomography(2,2) = 1;

	mMatches.clear();
	for(int i=0,end=mLinker.getMatches().size(); i!=end; ++i)
	{
		mMatches.emplace_back();
		mMatches.back().match = mLinker.getMatches()[i];
		mMatches.back().isInlier = mLinker.getInliers()[i];
		mMatches.back().isWithFrameA = &mMatches.back().match.sourceMeasurement->getKeyFrame() == &frameA;
	}

	if(mShowWarp)
		mDisplayText << "Overlaying warp\n";
}

void FrameLinkWindow::draw()
{
	TwoFrameWindow::draw();

	if(!mValidFrameA || !mValidFrameB)
		return;

	//Frame A
	mTiler.setActiveTile(0);
	mShaders->getColor().setMVPMatrix(mTiler.getMVP());
	mShaders->getTextureWarp().setMVPMatrix(mTiler.getMVP());
	if(mValidFrameA)
	{
		if(mShowWarp)
			mShaders->getTextureWarp().renderTexture(mFrameBTexture.getTarget(), mFrameBTexture.getId(), mHomography, 0.5f, mFrameBTexture.getSize());

		std::vector<std::vector<cv::Point2f>> points;
		std::vector<std::vector<cv::Vec4f>> colors;
		points.resize(mOctaveCount);
		colors.resize(mOctaveCount);
		for(auto &data : mMatches)
		{
			if(!data.isWithFrameA)
				continue;

			int octave = data.match.sourceMeasurement->getOctave();
			cv::Vec4f color;
			if(data.isInlier)
				color = StaticColors::Blue();
			else
				color = StaticColors::Red();
			points[octave].push_back(data.match.sourceMeasurement->getPositions()[0]);
			colors[octave].push_back(color);
		}
		for(int octave=0; octave<(int)points.size(); ++octave)
		{
			int scale = 1<<octave;
			mShaders->getColor().drawRect(points[octave].data(), colors[octave].data(), points[octave].size(), (float)(scale*PatchWarper::kPatchSize),1.0f);
		}
	}

	//Frame B
	mTiler.setActiveTile(1);
	mShaders->getColor().setMVPMatrix(mTiler.getMVP());
	if(mValidFrameB)
	{
		std::vector<std::vector<cv::Point2f>> points;
		std::vector<std::vector<cv::Vec4f>> colors;
		points.resize(mOctaveCount);
		colors.resize(mOctaveCount);
		for(auto &data : mMatches)
		{
			int octave = data.match.sourceMeasurement->getOctave();
			cv::Vec4f color;
			if(!data.isInlier)
				color = StaticColors::Red();
			else if(data.isWithFrameA)
				color = StaticColors::Blue();
			else
				color = StaticColors::Purple();

			points[octave].push_back(data.match.measurement.getPositions()[0]);
			colors[octave].push_back(color);
		}
		for(int octave=0; octave<(int)points.size(); ++octave)
		{
			int scale = 1<<octave;
			mShaders->getColor().drawRect(points[octave].data(), colors[octave].data(), points[octave].size(), (float)(scale*PatchWarper::kPatchSize),1.0f);
		}
	}
}

void FrameLinkWindow::runRegionLink()
{
	std::lock_guard<shared_mutex> lock(mSlam->getMap().getMutex());

	if(mSlam->getMap().getRegions().size() < 2)
	{
		DTSLAM_LOG << "Need at least 2 regions to link.\n";
		return;
	}

	auto &regionA = *mSlam->getMap().getRegions()[0];
	auto &regionB = *mSlam->getMap().getRegions()[1];

	//std::unordered_map<SlamKeyFrame *, std::unique_ptr<FrameLinkData>> links;
	std::vector<std::unique_ptr<FrameLinkData>> links;

	//Check for duplicate frame in both regions
	if(regionB.getPreviousRegionSourceFrame())
	{
		DTSLAM_LOG << "Previous!\n";

		std::unique_ptr<FrameLinkData> linkData(new FrameLinkData());
		linkData->frameA = regionB.getPreviousRegionSourceFrame();
		linkData->frameB = regionB.getKeyFrames().front().get();
		linkData->pose = regionB.getPreviousRegionSourceFrame()->getPose();
		for(auto &m : regionB.getPreviousRegionSourceFrame()->getMeasurements())
		{
			linkData->matches.push_back(FeatureMatch(FeatureProjectionInfo(), m, linkData->frameB, m->getOctave(), m->getPositions(), m->getPositionXns(), 1));
		}
		links.push_back(std::move(linkData));
	}
	else
		DTSLAM_LOG << "No previous!\n";

	//Find other links
	//for(auto &frameAptr : regionA.getKeyFrames())
	//{
	//	auto &frameA = *frameAptr;
	//	for(auto &frameBptr : regionB.getKeyFrames())
	//	{
	//		auto &frameB = *frameBptr;
	//		if(links.find(&frameB)!=links.end())
	//			continue;

	//		std::unique_ptr<FrameLinkData> linkData;

	//		linkData = mLinker.findLink(frameA, frameB);
	//		if(linkData)
	//		{
	//			links.insert(std::make_pair(&frameB, std::move(linkData)));
	//		}
	//	}
	//}
	for (int frameAidx = 0, endA = 4; frameAidx != endA; ++frameAidx)
	{
		auto &frameA = *regionA.getKeyFrames()[frameAidx];

		uint maxCount = 0;
		std::unique_ptr<FrameLinkData> bestLinkData;

		for (int frameBidx = regionB.getKeyFrames().size() - 4, endB = regionB.getKeyFrames().size() - 1; frameBidx != endB; ++frameBidx)
		{
			auto &frameB = *regionB.getKeyFrames()[frameBidx];

			std::unique_ptr<FrameLinkData> linkData;

			linkData = mLinker.findLink(frameA, frameB);
			if (linkData && linkData->matches.size() > maxCount)
			{
				maxCount = linkData->matches.size();
				bestLinkData = std::move(linkData);
			}
		}

		if (bestLinkData)
		{
			links.push_back(std::move(bestLinkData));
		}
	}

	DTSLAM_LOG << "Link count: " << links.size() << "\n";
	int i=1;
	for(auto &linkPtr : links)
	{
		auto &link = *linkPtr;
		DTSLAM_LOG << "\n" << "frameA(" << i << ")=" << link.frameA->getTimestamp() << "\n";
		DTSLAM_LOG << "frameB(" << i << ")=" << link.frameB->getTimestamp() << "\n";
		DTSLAM_LOG << "matchCount(" << i << ")=" << link.matches.size() << "\n";
		DTSLAM_LOG << "Ra{" << i << "}=" << (cv::Mat)link.pose.getRotationRef() << "\n";
		DTSLAM_LOG << "Ta{" << i << "}=" << link.pose.getTranslationRef() << "\n";
		DTSLAM_LOG << "Rb{" << i << "}=" << (cv::Mat)link.frameB->getPose().getRotation() << "\n";
		DTSLAM_LOG << "Tb{" << i << "}=" << link.frameB->getPose().getTranslation() << "\n";
		DTSLAM_LOG << "Rc{" << i << "}=" << (cv::Mat)link.frameA->getPose().getRotation() << "\n";
		DTSLAM_LOG << "Tc{" << i << "}=" << link.frameA->getPose().getTranslation() << "\n";

		//Add matches
		for(FeatureMatch &match : link.matches)
		{
			std::unique_ptr<SlamFeatureMeasurement> measurement(new SlamFeatureMeasurement(match.measurement));

			link.frameB->getMeasurements().push_back(measurement.get());
			measurement->getFeature().getMeasurements().push_back(std::move(measurement));
		}

		++i;
	}

	auto &link0 = *links.front();
	cv::Matx33f Rrel = link0.frameB->getPose().getRotation() * link0.pose.getRotationRef().t();
	cv::Vec3f meanA(0,0,0);
	cv::Vec3f meanB(0,0,0);
	std::vector<std::pair<cv::Vec3f,cv::Vec3f>> centers;
	for(auto &p : links)
	{
		auto &link = *p;

		centers.push_back(std::make_pair(
				link.pose.getCenter(),
				Rrel * link.frameB->getPose().getCenter())); //first=centerA, second=Rrel*centerB

		meanA += centers.back().first;
		meanB += centers.back().second;
	}
	meanA *= 1.0f/centers.size();
	meanB *= 1.0f/centers.size();

	float scale = 0;
	for(auto &c : centers)
	{
		cv::Vec3f a = c.first - meanA;
		cv::Vec3f b = c.second - meanB;
		scale += sqrtf(cvutils::NormSq(a)) / sqrtf(cvutils::NormSq(b));
	}
	scale /= centers.size();

	cv::Vec3f trel = meanA - scale*meanB;

	DTSLAM_LOG << "Rrel=" << (cv::Mat)Rrel << "\n";
	DTSLAM_LOG << "scale=" << scale << "\n";
	DTSLAM_LOG << "trel=" << trel << "\n";

	//Update region
//	for(auto &featurePtr : regionB.getFeatures3D())
//	{
//		auto &feature = *featurePtr;
//		feature.setPosition(scale*Rrel*feature.getPosition() + cv::Point3f(trel));
//	}
//	for(auto &framePtr : regionB.getKeyFrames())
//	{
//		auto &frame= *framePtr;
//		Pose3D *pose = &frame.getPose();
//		FullPose3D *fullpose = dynamic_cast<FullPose3D*>(pose);
//		if(fullpose)
//		{
//			cv::Matx33f newR = fullpose->getRotationRef() * Rrel.t();
//			cv::Vec3f newT = -newR * trel + scale*fullpose->getTranslationRef();
//			fullpose->set(newR, newT);
//		}
//	}

	//Join regions
	mSlam->getMap().mergeRegions(regionA, regionB, Rrel, scale, trel);
	if(mSlam->getActiveRegion() == &regionB)
		mSlam->setActiveRegion(&regionA);
}

} /* namespace dtslam */
