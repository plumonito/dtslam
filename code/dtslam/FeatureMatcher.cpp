/*
 * FeatureMatcher.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "FeatureMatcher.h"
#include "Profiler.h"
#include "log.h"
#include "PatchWarper.h"
#include "MatchRefiner.h"

#include "flags.h"

namespace dtslam
{
void FeatureMatch::writeToMatlabLog() const
{
	{
		auto &m = *sourceMeasurement;
		auto &frame = m.getKeyFrame();
		MatlabDataLog::AddCell("refPoseR", (cv::Mat)frame.getPose().getRotation());
		MatlabDataLog::AddCell("refPoseT", frame.getPose().getTranslation());
		MatlabDataLog::AddCell("refPos", m.getPositions()[0]);
		MatlabDataLog::AddCell("refXn", m.getPositionXns()[0]);
	}
	{
		auto &m = measurement;
		auto &frame = m.getKeyFrame();
		MatlabDataLog::AddCell("imgPoseR", (cv::Mat)frame.getPose().getRotation());
		MatlabDataLog::AddCell("imgPoseT", frame.getPose().getTranslation());
		MatlabDataLog::AddCell("imgPos", m.getPositions()[0]);
		MatlabDataLog::AddCell("imgXn", m.getPositionXns()[0]);
	}
}

void MatchingResultsData::createMatchMap()
{
	mMatchedProjectionMap.clear();
	for(auto &m : mMatches2D)
	{
		mMatchedProjectionMap.insert(std::make_pair(&m.projection.getFeature(), &m));
	}
	for (auto &m : mMatches3D)
	{
		mMatchedProjectionMap.insert(std::make_pair(&m.projection.getFeature(), &m));
	}
}

void MatchingResultsData::resort2D3D()
{
	std::vector<FeatureMatch> matches2D;

	for(auto &match : mMatches2D)
	{
		if (match.projection.getFeature().is3D())
		{
			mMatches3D.push_back(match);
		}
		else
		{
			matches2D.push_back(match);
		}
	}

	mMatches2D = std::move(matches2D);
	createMatchMap();
}

FeatureMatcher::FeatureMatcher()
{
	mNonMaximaPixelSize = FLAGS_MatcherNonMaximaPixelSize;
	mMaxZssdScore = FLAGS_MatcherMaxZssdScore;
	mBestScorePercentThreshold = (float)FLAGS_MatcherBestScorePercentThreshold;
}

void FeatureMatcher::clearResults()
{
	mMatchAttempts.clear();
	mMatches.clear();
}

void FeatureMatcher::setFrame(const SlamKeyFrame *frame)
{
	mFrame = frame;

	mUseAffine = false;

	int octaveCount = frame->getOctaveCount();
	mTransformedKeypoints.clear();
	mTransformedKeypoints.resize(octaveCount);
	for (int octave = 0; octave != octaveCount; ++octave)
	{
		auto &transformedPositions = mTransformedKeypoints[octave];
		transformedPositions.create(frame->getPyramid()[octave].size(), cv::Size2i(20, 20), 0);
		for (auto &kp : frame->getKeyPoints(octave))
		{
			TransformedKeypointPosition pos;
			pos.original = kp.position;
			pos.transformed = kp.position;
			pos.transformedXn = kp.xn;

			transformedPositions.addFeature(pos);
		}
	}
}

void FeatureMatcher::setFrame(const SlamKeyFrame *frame, const cv::Matx23f &keypointToProjection)
{
	mFrame = frame;

	mUseAffine = true;
	mKeypointToProjectionAffine = keypointToProjection;

	int octaveCount = frame->getOctaveCount();
	mTransformedKeypoints.clear();
	mTransformedKeypoints.resize(octaveCount);
	for (int octave = 0; octave != octaveCount; ++octave)
	{
		auto &transformedPositions = mTransformedKeypoints[octave];
		transformedPositions.create(frame->getPyramid()[octave].size(), cv::Size2i(20,20), 0);
		for (auto &kp : frame->getKeyPoints(octave))
		{
			TransformedKeypointPosition pos;
			pos.transformed = cvutils::AffinePoint(mKeypointToProjectionAffine, kp.position);
			if (pos.transformed.x < 0 || pos.transformed.x >= mCamera->getImageSize().width ||
				pos.transformed.y< 0 || pos.transformed.y >= mCamera->getImageSize().height)
				continue;

			pos.original = kp.position;
			pos.transformedXn = mCamera->unprojectToWorldLUT(pos.transformed);
			transformedPositions.addFeature(pos);
		}
	}
}

void FeatureMatcher::setFramePose(const Pose3D &pose)
{
	mPose.set(pose);
}

int FeatureMatcher::findMatches(const std::vector<FeatureProjectionInfo> &projectionsToMatch)
{
	int searchCount = projectionsToMatch.size();
	int foundMatchesCount=0;
	int foundKeyPointsCount=0;

	DTSLAM_LOG << "Searching for " << (searchCount) << " features...\n";
	auto startt = std::chrono::high_resolution_clock::now();

	for(auto &projection : projectionsToMatch)
	{
		FeatureMatch *match;
		match = findMatch(projection);
		if(match)
		{
			foundMatchesCount++;
			foundKeyPointsCount += match->measurement.getPositionCount();
		}
	}

	auto endt = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endt - startt);

	float percentage = (foundMatchesCount) ? (100*foundKeyPointsCount / (float)foundMatchesCount) : 0.0f;
	DTSLAM_LOG << "Found " << (foundMatchesCount) << " matches (" << foundKeyPointsCount << " key points, " << percentage << "%), took " << duration.count() << "ms.\n";
	return foundMatchesCount;
}

FeatureMatch * FeatureMatcher::findMatch(const FeatureProjectionInfo &projection)
{
	//ProfileSection s("findMatch");

	std::vector<cv::Point2f> positions;
	std::vector<cv::Point3f> positionXns;

	//We can only match with measurements that have a single point
	assert(projection.getSourceMeasurement().getPositions().size()==1);

	bool found;

	//Log attempt
	//TODO: logging this is good for debugging and analyzing matcher performance but might be a significant performance hit.
	auto it = mMatchAttempts.insert(std::make_pair(&projection.getFeature(), MatchAttempt()));
	auto &attempt = it.first->second;
	attempt.projection = projection;
	std::vector<MatchCandidate> &candidates = attempt.candidates;

	//Search
	found = findMatch(projection, candidates, positions, positionXns);

	//Add if found
	if(found)
	{
		mMatches.emplace_back(projection, &projection.getSourceMeasurement(), &getFrame(), projection.getOctave(), positions, positionXns, projection.getTrackLength());
		return &mMatches.back();
	}
	else
		return nullptr;
}

bool FeatureMatcher::findMatch(const FeatureProjectionInfo &projection,
		std::vector<MatchCandidate> &candidates,
		std::vector<cv::Point2f> &positions,
		std::vector<cv::Point3f> &positionXns)
{
	//ProfileSection s("findMatchSingle");
	assert(mCamera);
	assert(mFrame);

	//auto &projection = attempt.projection;

	//Set source on warper
	PatchWarper &warper = mWarperCache.get(&projection.getFeature());
	const SlamFeatureMeasurement &m = projection.getSourceMeasurement();
	{
		//ProfileSection s("warperSetSource");
		warper.setSource(&m.getKeyFrame().getCameraModel(), &m.getImage(), m.getOctave(), m.getUniquePosition(), m.getUniquePositionXn());
	}


	//Search according to projection type
	std::vector<cv::Point2f> searchPositions;
	cv::Point2f warpPosition;
	{
		//ProfileSection s("candidates");
		if(projection.getType() == EProjectionType::EpipolarLine)
		{
			auto &epipolarData = projection.getEpipolarData();
			warpPosition = mCamera->projectFromWorld(epipolarData.infiniteXn);
			getEpipolarCandidates(getFrame(), projection.getOctave(), epipolarData.epiPlaneNormal, epipolarData.minDepthXn, epipolarData.infiniteXn, warpPosition, searchPositions);
		}
		else
		{
			auto &pointData = projection.getPointData();
			warpPosition = pointData.positions[0];
			getPointCandidates(getFrame(), projection.getOctave(), pointData.positions, searchPositions);

			//Add the original position in case the feature detector missed it
			for (auto &pos : pointData.positions)
				searchPositions.push_back(pos);
		}
	}

	//Extract patch from reference image
	{
		//ProfileSection s("updateWarp");
		cv::Matx33f relR = m.getKeyFrame().getPose().getRotation() * getPose().getRotationRef().t(); //TODO: cache r.t()
		const int scale = 1 << projection.getOctave();
		if(mUseAffine)
			warper.calculateWarp(relR, *mCamera, mKeypointToProjectionAffine, warpPosition, scale);
		else
			warper.calculateWarp(relR, *mCamera, warpPosition, scale);

		if(warper.patchNeedsUpdate())
		{
			warper.updatePatch();
		}
		if(warper.getPatch().empty())
			return false;
	}

	//attempt.refPatch = warper.getPatch();

	return findMatch(warper.getPatch(), projection.getOctave(), searchPositions, candidates, positions, positionXns);
}

void FeatureMatcher::getPointCandidates(const SlamKeyFrame &frame,
		const int octave,
		const std::vector<cv::Point2f> &searchCenters,
		std::vector<cv::Point2f> &candidates)
{
	//ProfileSection s("point");
	assert(mCamera);
	assert(mFrame);

	//Find candidates
	//for (auto &center : searchCenters)
	//{
	//	std::vector<TransformedKeypointPosition *> vec;
	//	mTransformedKeypoints[octave].getFeaturesInRect(vec, cv::Rect2i(center.x-mSearchPixelDistance, center.y-mSearchPixelDistance, 2*mSearchPixelDistance, 2*mSearchPixelDistance));
	//	for (auto p : vec)
	//		candidates.push_back(p->original);
	//}

	for(auto &pos: mTransformedKeypoints[octave])
	{
		for (auto &center : searchCenters)
		{
			const int diffX = abs((int)(pos.transformed.x - center.x));
			const int diffY = abs((int)(pos.transformed.y - center.y));
			if (diffX <= mSearchPixelDistance && diffY <= mSearchPixelDistance)
			{
				candidates.push_back(pos.original);
				break;
			}
		}
	}
}

void FeatureMatcher::getEpipolarCandidates(const SlamKeyFrame &frame,
	const int octave,
	const cv::Vec3f &epiPlaneNormal,
	const cv::Point3f &epipole,
	const cv::Point3f &infiniteXn,
	const cv::Point2f &infiniteUv,
	std::vector<cv::Point2f> &candidates)
{
	//ProfileSection s("epipolar");
	assert(mCamera);
	//Epipolar geometry
	cv::Vec3f lineDir = infiniteXn - epipole;

	auto &camera = frame.getCameraModel();
	cv::Point2i jacobianUv((int)infiniteUv.x, (int)infiniteUv.y);
	if (jacobianUv.x < 0)
		jacobianUv.x = 0;
	else if (jacobianUv.x >= camera.getImageSize().width)
		jacobianUv.x = camera.getImageSize().width - 1;
	if (jacobianUv.y< 0)
		jacobianUv.y= 0;
	else if (jacobianUv.y >= camera.getImageSize().height)
		jacobianUv.y = camera.getImageSize().height - 1;
	cv::Vec3f ujac;
	cv::Vec3f vjac;
	//camera.projectFromWorldJacobian(infiniteXn, ujac, vjac);
	camera.projectFromWorldJacobianLUT(jacobianUv, ujac, vjac);

	//Find candidates
	for (auto &keypoint : mTransformedKeypoints[octave])
	{
		//Check that it is not too far away from the epipolar segment
		cv::Vec3f distXn;

		//Calculate distance to infinite point
		distXn = keypoint.transformedXn-infiniteXn;

		//Check if it is beyond infinite
		float signA = lineDir.dot(distXn);
		if(signA < 0)
		{
			//Not beyond
			//Calculate distance to min depth point
			distXn = keypoint.transformedXn - epipole;

			//Now check if it is before min depth
			float signB = lineDir.dot(distXn);
			if(signB > 0)
			{
				//Not beyond
				//Calculate distance to the epipolar plane
				float distanceToPlane = epiPlaneNormal.dot(keypoint.transformedXn);
				distXn = epiPlaneNormal*distanceToPlane;
			}
		}

		//Now convert to pixel units
		float du = ujac.dot(distXn);
		float dv = vjac.dot(distXn);

		float distSq = du*du + dv*dv;

		if(distSq < mSearchPixelDistanceSq)
		{
			candidates.push_back(keypoint.original);
		}
	}
}

bool FeatureMatcher::findMatch(const cv::Mat1b &refPatch,
		const int octave,
		const std::vector<cv::Point2f> &searchPositions,
		std::vector<MatchCandidate> &candidates,
		std::vector<cv::Point2f> &positions,
		std::vector<cv::Point3f> &positionXns)
{
	//ProfileSection s("findMatchFinal");
	const int scale = 1<<octave;

	int bestScore=10*mMaxZssdScore;

	candidates.resize(searchPositions.size());

	//Prepare refiner
	MatchRefiner refiner;
	refiner.setRefPatch(&refPatch);
	refiner.setImg(&getFrame().getImage(octave));

	//Match
	std::vector<MatchCandidate *> initialMatches;
	for(int i=0,end=searchPositions.size(); i!=end; ++i)
	{
		auto &searchPos = searchPositions[i];
		auto &candidate = candidates[i];

		candidate.initialPos = searchPos;
		candidate.isMatch = false;

		//Refine
		cv::Point2f initialCenter = cv::Point2f(searchPos.x / scale, searchPos.y / scale);
		refiner.setCenter(initialCenter);
		if(!refiner.refine() || cvutils::PointDistSq(initialCenter, refiner.getCenter()) > 10*10)
		{
			candidate.score = std::numeric_limits<int>::max();
			continue;
		}

		candidate.refinedPos = refiner.getCenter()*scale;
		//candidate.refinedPatch = refiner.getImgPatch();
		candidate.score = refiner.getScore();

		if(candidate.score < mMaxZssdScore)
		{
			//Passes the score, add
			candidate.isMatch = true;
			initialMatches.push_back(&candidate);

			//Track best
			if(candidate.score < bestScore)
			{
				bestScore = candidate.score;
			}
		}
	}

	//Add to result
	int maxScore = (int)(bestScore*FLAGS_MatcherBestScorePercentThreshold);
	for(int i=0,end=initialMatches.size(); i!=end; ++i)
	{
		auto &match = *initialMatches[i];

		//Ignore match if score is above threshold
		if(match.score > maxScore)
		{
			match.isMatch = false;
		}
		else
		{
			//Determine whether it is a local maxima
			bool isMaxima=true;
			for(int j=0; j!=end; ++j)
			{
				if(i == j)
					continue;

				auto &match2 = *initialMatches[j];
				if(fabs(match.refinedPos.x - match2.refinedPos.x) <= mNonMaximaPixelSize && fabs(match.refinedPos.y - match2.refinedPos.y) <= mNonMaximaPixelSize)
				{
					if((match.score > match2.score) || (match.score == match2.score && i>j))
					{
						isMaxima = false;
						break;
					}
				}
			}

			//Ignore match is not a local maxima
			if(!isMaxima)
			{
				match.isMatch = false;
			}
		}

		if(match.isMatch)
		{
			//Add to match list
			positions.push_back(match.refinedPos);
			positionXns.push_back(getFrame().getCameraModel().unprojectToWorld(match.refinedPos));
		}
	}

//	if(positions.empty())
//	{
//		DTSLAM_LOG << "AHHH NOT FOUND!\n";
//	}
	return !positions.empty();
}

} /* namespace dtslam */
