/*
 * FrameLinkWindow.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef FRAMELINKWINDOW_H_
#define FRAMELINKWINDOW_H_

#include "TwoFrameWindow.h"
#include "dtslam/FeatureMatcher.h"
#include "dtslam/FrameLinker.h"

namespace dtslam
{

class FrameLinkWindow: public TwoFrameWindow
{
public:
	FrameLinkWindow():
		TwoFrameWindow("FrameLinkWindow"), mShowWarp(true)
	{}
	~FrameLinkWindow();

	bool init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize);
	void showHelp() const;

	void draw();

protected:
	FrameLinker mLinker;

	int mOctaveCount;

	bool mLinkSuccesfull;
	cv::Matx33f mHomography;

	struct MatchData
	{
		FeatureMatch match;
		bool isInlier;
		bool isWithFrameA;
	};
	std::vector<MatchData> mMatches;

	bool mShowWarp;

	void updateState(const SlamKeyFrame &frameA, const SlamKeyFrame &frameB);
	void toggleShowWarp();
	void runRegionLink();
};

} /* namespace dtslam */

#endif /* FRAMELINKWINDOW_H_ */
