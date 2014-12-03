/*
 * MatchesWindow.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef MATCHESWINDOW_H_
#define MATCHESWINDOW_H_

#include <array>
#include "BaseWindow.h"
#include "dtslam/PoseTracker.h"

namespace dtslam {

class MatchesWindow: public BaseWindow
{
public:
	MatchesWindow():
		BaseWindow("MatchesWindow"),
		mSelectedFeature(NULL),
		mDrawSelectedProjectionTriangle(false),
		mDrawSelectedProjectionEpiLine(false)
	{}

	bool init(SlamDriver *app, SlamSystem *slam, const cv::Size &imageSize);
    void showHelp() const;

    void updateState();

    void touchDown(int id, int x, int y);

    void resize();
    void draw();

protected:
    ViewportTiler mTiler;
	TextureHelper mRefFrameTexture;

    const SlamFeature *mSelectedFeature; // Selected feature, mSelectedMatchAttempt
	const MatchAttempt *mSelectedMatchAttempt;
	float mSelectedMinScore;

    MatchReprojectionErrors mSelectedErrors;

    cv::Matx23f mProjectionToKeyPointAffine;

    //Drawing stuff
    FullPose3D mFramePose;
    CameraModel mCamera;

	std::vector<std::vector<cv::Point2f>> mSquareCenters;
	std::vector<std::vector<cv::Vec4f>> mSquareColors;

	std::vector<std::string> mDrawText;
	std::vector<cv::Point2f> mDrawTextPos;
	std::vector<cv::Vec4f> mDrawTextColor;

	std::vector<std::array<cv::Point2f,2>> mLines;
	std::vector<cv::Vec4f> mLineColors;

	std::vector<cv::Point2f> mTriangleCenters;

    std::vector<std::vector<cv::Point2f>> mEpiLines;

    bool mDrawSelectedProjectionTriangle;
    cv::Point2f mSelectedProjectionTriangleCenter;

    bool mDrawSelectedProjectionEpiLine;
    std::vector<cv::Point2f> mSelectedProjectionEpiLine;

    bool mDrawRefFrame;
    cv::Point2f mRefMeasurementPos;
    int mRefMeasurementOctave;

    //Key binding functions
    void increaseDebugMatchIdx();
    void decreaseDebugMatchIdx();
    void resetDebugMatchIdx();
	void executeRefiner();

	cv::Vec4f selectMatchColor(const FeatureMatch &match, const MatchReprojectionErrors &errors);
	void drawMatch(const FeatureMatch &match, const MatchReprojectionErrors &error);
};

} /* namespace dtslam */

#endif /* MATCHESWINDOW_H_ */
