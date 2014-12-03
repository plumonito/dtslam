/*
 * PoseTracker.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "PoseTracker.h"

#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "Profiler.h"
#include "log.h"
#include "HomographyEstimation.h"
#include "Pose3D.h"
#include "FeatureMatcher.h"
#include "RotationEstimation.h"
#include "EssentialEstimation.h"
#include "PnpEstimation.h"
#include "PoseEstimation.h"

#include "flags.h"

namespace dtslam {

void PoseTracker::init(const CameraModel *camera, const cv::Size &imageSize, int octaveCount)
{
	mCamera = camera;
	mImageSize = imageSize;
	mOctaveCount = octaveCount;

	//mIsLost = false;

	//Translate search radius to normalized units
	mMatcherSearchRadius = (1<<(octaveCount-1)) * FLAGS_MatcherPixelSearchDistance;
	DTSLAM_LOG << "PoseTracker: Matcher search radius = " << mMatcherSearchRadius << "px in image res\n";

	mFeatureSortGridSize = cv::Size2i(5,5);

	//Matcher
	mMatcher.reset(new FeatureMatcher());
	mMatcher->setCamera(mCamera);

	//Model refiners
	mHomographyEstimator.reset(new HomographyEstimation());

	mPoseEstimator.reset(new PoseEstimator());
	mPoseEstimator->init(mCamera, (float)FLAGS_TrackerOutlierPixelThreshold);
}

const FeatureMatch *PoseTracker::getMatch(const SlamFeature *feature) const
{
	auto it = mMatchMap.find(feature);
	if (it == mMatchMap.end())
		return NULL;
	else
		return it->second;
}

void PoseTracker::resetTracking(const Pose3D &initialPose)
{
	mCurrentPose.set(initialPose);
	mPoseType = EPoseEstimationType::FullPose;

	//Forget all previous matches
	mLastFrame.reset(NULL);
	mLastMatches.clear();

	//Forget current frame
	mFrame.reset(NULL);
	mFeaturesInView.clear();
	mFeaturesInView.resize(mOctaveCount);
	mMatchAttempts.clear();
	mMatches.clear();
	mMatchMap.clear();
	mReprojectionErrors.clear();
}

bool PoseTracker::trackFrame(std::unique_ptr<SlamKeyFrame> frame_)
{
	ProfileSection s("trackFrame");

	//Save old data 
	mLastFrame.reset(NULL);
	mLastMatches.clear();
	if (mFrame)
	{
		mLastFrame = std::move(mFrame);

		for (int i = 0, end = mMatches.size(); i != end; ++i)
		{
			if (mReprojectionErrors[i].isInlier)
			{
				mLastMatches.push_back(mMatches[i]);
			}
		}
	}

	//Reset new frame data
	mFrame = std::move(frame_);
	mFeaturesInView.clear();
	mFeaturesInView.resize(mOctaveCount);
	mMatchAttempts.clear();
	mMatches.clear();
	mMatchMap.clear();
	mReprojectionErrors.clear();

	///////////////////////////////////////////////////////////////////////////////////
	//Estimate initial similarity
	//Small blurry image
	if (estimateSimilarityFromLastFrame(*mFrame, mSimilarity))
	{
		cv::invertAffineTransform(mSimilarity, mSimilarityInv);
		
		//Prepare matcher
		mMatcher->setFrame(mFrame.get(), mSimilarityInv);
	}
	else
	{
		mSimilarity = mSimilarityInv = cv::Matx23f::eye();

		//Prepare matcher
		mMatcher->setFrame(mFrame.get());
	}
	mMatcher->setFramePose(getCurrentPose());

	mMatcher->setSearchDistance(mMatcherSearchRadius);

	///////////////////////////////////////////////////////////////////////////////////
	//Select features to match

	//Remember previous matches so we don't project them
	std::unordered_set<SlamFeature*> featuresToIgnore;
	if (mLastFrame)
	{
		for (auto &match : mLastMatches)
		{
			featuresToIgnore.insert(&match.projection.getFeature());
		}
	}

	//Find features in view
	mActiveRegion->getFeaturesInView(
			getCurrentPose(),
			mFrame->getCameraModel(),
			getOctaveCount(),
			true,
			featuresToIgnore,
			mFeaturesInView);
	//Debug: Show count of the number of features
	{
		std::vector<int> featureInViewCount2D(getOctaveCount(),0);
		std::vector<int> featureInViewCount3D(getOctaveCount(), 0);
		for (int octave = getOctaveCount() - 1; octave >= 0; --octave)
		{
			for (auto &projection : mFeaturesInView[octave])
			{
				if (projection.getFeature().is3D())
					featureInViewCount3D[octave]++;
				else
					featureInViewCount2D[octave]++;
			}
		}
		DTSLAM_LOG << "3D Features in view by octave: " << featureInViewCount3D << ".\n";
		DTSLAM_LOG << "2D Features in view by octave: " << featureInViewCount2D << ".\n";
	}

	//Add all matches from the previous frame
	std::vector<std::vector<FeatureProjectionInfo>> previousMatches;
	previousMatches.resize(mOctaveCount);
	if(mLastFrame)
	{
		for (auto &match : mLastMatches)
		{
			auto &m = match.measurement;
			
			if (m.getFeature().getStatus() == SlamFeatureStatus::Invalid)
				continue; //Ignore features that have moved to the garbage

			previousMatches[m.getOctave()].push_back(FeatureProjectionInfo::CreatePreviousMatch(&m.getFeature(), const_cast<SlamFeatureMeasurement*>(match.sourceMeasurement), m.getOctave(), match.trackLength + 1, m.getPositions()));
		}
	}


	///////////////////////////////////////////////////////////////////////////////////
	//Do first pass (using the help of the similarity)
	//For now we do only one pass. PTAM does two passes, one with the help of the similarity for low resolution features and one for high resolution features
	{
		//ProfileSection s("Pass1");

		//Go through octaves
		int featuresNeeded = FLAGS_TrackerMaxFeatures;

		{
			ProfileSection s("matching");

			for(int octave=mOctaveCount-1; octave>=0; --octave)
			{
				int totalInOctaveCount = (int)(mFeaturesInView[octave].size() + previousMatches[octave].size());
				int featuresNeededOctave = std::min(featuresNeeded, std::min(totalInOctaveCount, FLAGS_TrackerMaxFeaturesPerOctave));
				std::vector<TSortedFeaturesCell> featureGrid;
				std::vector<FeatureProjectionInfo*> features2D;

				//Sort and match
				sortFeaturesInOctave(previousMatches[octave], mFeaturesInView[octave], featureGrid, features2D);
				matchFeaturesInOctave(featuresNeededOctave, featureGrid, features2D);

				featuresNeeded -= featuresNeededOctave;
				if(featuresNeeded <= 0)
					break;
			}
		}

		//Get matching results
		mMatchAttempts = std::move(mMatcher->getMatchAttempts());
		mMatches = std::move(mMatcher->getMatches());
		
		//Create the match map
		for (auto &match : mMatches)
		{
			mMatchMap.insert(std::make_pair(&match.measurement.getFeature(), &match));
		}

		//Debug: show match count
		int foundMatchCount=0, foundKeyPointCount=0;
		std::vector<int> foundCountByOctave(mOctaveCount,0);

		foundMatchCount = mMatches.size();
		int count3D = 0;
		int count2D = 0;
		for (auto &match : mMatches)
		{
			if (match.measurement.getFeature().is3D())
				count3D++;
			else
				count2D++;
			foundCountByOctave[match.measurement.getOctave()]++;
			foundKeyPointCount += match.measurement.getPositionCount();
		}
		float percentage = (foundMatchCount) ? (100*foundKeyPointCount / (float)foundMatchCount) : 0.0f;
		DTSLAM_LOG << "Found " << (foundMatchCount) << " matches (" << count3D << " 3D + " << count2D << " 2D) with " << percentage << "% key points";
		for(int octave=mOctaveCount-1; octave>=0; --octave)
			DTSLAM_LOG << ", octave " << octave << "=" << foundCountByOctave[octave];
		DTSLAM_LOG << "\n";

		if (foundMatchCount < 5)
		{
			DTSLAM_LOG << "Not enough matches, aborting tracking.\n";
			mPoseType = EPoseEstimationType::Invalid;
			mMatchInlierCount = 0;
			mEssentialReferenceFrame = NULL;
			mReprojectionErrors.resize(mMatches.size());
			mReprojectionErrorsForRotation.resize(mMatches.size());
		}
		else
		{
			//Estimate pose
			mPoseEstimator->setPreviousPose(getCurrentPose());
			mPoseEstimator->fitModels(mMatches);

			if (mPoseEstimator->getPoseType() == EPoseEstimationType::Invalid)
			{
				DTSLAM_LOG << "Pose estimation failed, aborting tracking.\n";
				mPoseType = EPoseEstimationType::Invalid;
				mMatchInlierCount = 0;
				mEssentialReferenceFrame = NULL;
				mReprojectionErrors.resize(mMatches.size());
				mReprojectionErrorsForRotation.resize(mMatches.size());
			}
			else
			{
				mPoseType = mPoseEstimator->getPoseType();
				mCurrentPose.set(mPoseEstimator->getPose());
				mEssentialReferenceFrame = mPoseEstimator->getEssentialReferenceFrame();

				//Copy residuals
				mMatchInlierCount = mPoseEstimator->getInlierCount();
				mReprojectionErrors = std::move(mPoseEstimator->getReprojectionErrors());
				mReprojectionErrorsForRotation = std::move(mPoseEstimator->getPureRotationReprojectionErrors());
			}
		}
	}

	mFrame->setPose(std::unique_ptr<Pose3D>(new FullPose3D(getCurrentPose())));

	return mMatchInlierCount > 20;
}

bool PoseTracker::estimateSimilarityFromLastFrame(const SlamKeyFrame &frame, cv::Matx23f &similarity)
{
	similarity = cv::Matx23f::eye(); //Identity is used as the initial guess
	if(mLastFrame)
	{
		ProfileSection ss("estimateSimilarity");
		bool res = mHomographyEstimator->estimateSimilarityDirect(mLastFrame->getSBI(), mLastFrame->getSBIdx(), mLastFrame->getSBIdy(), frame.getSBI(), similarity);

		//Scale similarity matrix
		//similarity = scaleUp*similarity*scaleDown
		const int scale = mImageSize.width / frame.getSBI().cols;
		similarity(0,2) *= scale;
		similarity(1,2) *= scale;
		return res;
	}
	else
	{
		return false;
	}
}

void PoseTracker::sortFeaturesInOctave(const std::vector<FeatureProjectionInfo> &previousMatches, 
		const std::vector<FeatureProjectionInfo> &featuresInOctave,
		std::vector<TSortedFeaturesCell> &featureGrid,
		std::vector<FeatureProjectionInfo*> &features2D)
{
	ProfileSection s("sortInOctave");
	const int cellCount = mFeatureSortGridSize.area();

	featureGrid.resize(cellCount);

	const int cellWidth = mImageSize.width / mFeatureSortGridSize.width;
	const int cellHeight = mImageSize.height / mFeatureSortGridSize.height;

	//Randomize
	std::vector<FeatureProjectionInfo*> randomProjections;
	for(auto &projection : featuresInOctave)
		randomProjections.push_back(const_cast<FeatureProjectionInfo*>(&projection));
	std::random_shuffle(randomProjections.begin(), randomProjections.end());

	//Separate into cells
	for(auto &projectionPtr : randomProjections)
	{
		auto &projection = *projectionPtr;

		if(projection.getType() == EProjectionType::EpipolarLine)
		{
			//Epipolar line
			features2D.push_back(&projection);
		}
		else
		{
			//Point projection
			//Add first projected positions
			const auto pos = projection.getPointData().positions[0];

			int celli = (int)(pos.x / cellWidth);
			if (celli < 0 || celli >= mFeatureSortGridSize.width)
				continue;

			int cellj = (int)(pos.y / cellHeight);
			if (cellj < 0 || cellj >= mFeatureSortGridSize.height)
				continue;

			int cellIdx = cellj*mFeatureSortGridSize.width + celli;

			featureGrid[cellIdx].push_back(&projection);
		}
	}

	//Add previous matches last
	for (auto &projection : previousMatches)
	{
		//Point projection
		//Add first projected positions
		const auto pos = projection.getPointData().positions[0];

		int celli = (int)(pos.x / cellWidth);
		if (celli < 0 || celli >= mFeatureSortGridSize.width)
			continue;

		int cellj = (int)(pos.y / cellHeight);
		if (cellj < 0 || cellj >= mFeatureSortGridSize.height)
			continue;

		int cellIdx = cellj*mFeatureSortGridSize.width + celli;

		featureGrid[cellIdx].push_back(const_cast<FeatureProjectionInfo*>(&projection));
	}
}

void PoseTracker::matchFeaturesInOctave(int maxFeatureCount,
		std::vector<TSortedFeaturesCell> &featureGrid,
		std::vector<FeatureProjectionInfo *> &features2D)
{
	ProfileSection s("matchFeaturesInOctave");
	const int cellWidth = mImageSize.width / mFeatureSortGridSize.width;
	const int cellHeight = mImageSize.height / mFeatureSortGridSize.height;

	int featuresNeeded = maxFeatureCount;

	//Keep track of match counts
	std::vector<FeatureProjectionInfo*> attempts;
	attempts.reserve(featuresNeeded);

	int totalSearchPoint = 0;
	int totalSearchEpipolar = 0;
	int totalFoundPoint = 0;
	int totalFoundEpipolar = 0;

	//Match point projections
	{
		ProfileSection ss("selectAttempts");

		//Valid cells ( = cells with features)
		std::list<TSortedFeaturesCell*> validCells; 
		for (auto &cell : featureGrid)
		{
			if (!cell.empty())
				validCells.push_back(&cell);
		}

		while (featuresNeeded>0 && !validCells.empty())
		{
			int validCellCount = validCells.size();
			int attemptsPerCell = (featuresNeeded / validCellCount)+1;

			for (auto validIt = validCells.begin(), validEnd=validCells.end(); validIt != validEnd;)
			{
				auto &cell = **validIt;

				//Match
				int attemptCount = std::min((int)cell.size(), attemptsPerCell);
				for (int i = 0; i != attemptCount; ++i)
				{
					attempts.push_back(cell.back());
					cell.pop_back();
				}
				totalSearchPoint += attemptCount;
				featuresNeeded -= attemptCount;

				//Exit if we've matched enough
				if (featuresNeeded < 1)
					break;

				//Advance cell (delete if empty)
				if (cell.empty())
				{
					validIt = validCells.erase(validIt);
					validEnd = validCells.end();
				}
				else
					++validIt;
			}
		}

		//Match epipolar projections

		//This avoids any 2D matches when PoseUse2D is not enabled
		if (FLAGS_PoseUse2D || mActiveRegion->getFeatures3D().size() < 50)
		{
			//Match epipolar features
			while (featuresNeeded > 0 && !features2D.empty())
			{
				attempts.push_back(features2D.back());
				features2D.pop_back();

				totalSearchEpipolar++;
				featuresNeeded--;
			}
		}
	}

	//Match
	{
		ProfileSection ss("matching");
		for (auto projectionPtr : attempts)
		{
			auto &projection = *projectionPtr;

			FeatureMatch *match;
			match = mMatcher->findMatch(projection);
			if (match)
			{
				//Matched!
				if (projection.getType() == EProjectionType::EpipolarLine)
					totalFoundEpipolar++;
				else
					totalFoundPoint++;
			}
		}
	}

	//DTSLAM_LOG << "Searching for " << (totalSearchPoint + totalSearchEpipolar) << " features (" << totalSearchPoint << " points + " << totalSearchEpipolar << " epipolar).\n";
	//DTSLAM_LOG << "Found " << (totalFoundPoint + totalFoundEpipolar) << " matches (" << totalFoundPoint << " points + " << totalFoundEpipolar << " epipolar).\n";
}

void PoseTracker::updateFeatureProjections(const Pose3D &pose,
		std::vector<FeatureProjectionInfo> &featureProjections)
{
	assert(false);
	for(int i=0; i<(int)featureProjections.size(); ++i)
	{
		FeatureProjectionInfo &projection = featureProjections[i];
		SlamFeature &feature = projection.getFeature();

		if(feature.is3D())
		{
			//Update projection of 3D feature
			//const cv::Point3f xc = pose.apply(feature.getPosition());
			//const cv::Point2f uv = mCamera->projectFromWorld(xc);

			//projection.setPosition(uv);
			//TODO Update warp for 3D feature
		}
		else
		{
			//TODO:Update projection of 2D feature
			//projection.position = SlamMap::Project2d(feature.getPosition(), pose.getRotation(), *mCamera);
		}
	}
}

void PoseTracker::resync()
{
	if(!mFrame || mPoseType == EPoseEstimationType::Invalid)
		return;

	//Refit position
	mPoseEstimator->fitModels(mMatches);
	
	mCurrentPose.set(mPoseEstimator->getPose());
	mFrame->setPose(std::unique_ptr<Pose3D>(new FullPose3D(getCurrentPose())));

	//Copy residuals
	mMatchInlierCount = mPoseEstimator->getInlierCount();
	mReprojectionErrors = std::move(mPoseEstimator->getReprojectionErrors());
	mReprojectionErrorsForRotation = std::move(mPoseEstimator->getPureRotationReprojectionErrors());
}

} /* namespace dtslam */
