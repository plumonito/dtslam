/*
 * HomographyReprojectionError.h
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef HOMOGRAPHYREPROJECTIONERROR_H_
#define HOMOGRAPHYREPROJECTIONERROR_H_

#include <opencv2/core.hpp>

namespace dtslam {

class HomographyEstimation
{
public:
	HomographyEstimation(): mMaxIterations(50), mShowIterations(false)
	{
	}

	const int &getMaxIterations() {return mMaxIterations;}
	void setMaxIterations(int value) {mMaxIterations=value;}

	const bool &getShowIterations() {return mShowIterations;}
	void setShowIterations(bool value) {mShowIterations=value;}

	////////////////////////////////////////////////
	// Full homography estimation based on matches
	////////////////////////////////////////////////

	cv::Matx33f estimate(const cv::Matx33f &initial, const std::vector<cv::Point2f> &left, const std::vector<cv::Point2f> &right, const std::vector<int> &octave, float threshold, std::vector<bool> &inliers)
	{
		return estimateCeres(initial, left, right, octave, threshold, inliers);
		//return EstimateOpenCV(initial, left, right, threshold, inliers);
	}
	cv::Matx33f estimateCeres(const cv::Matx33f &initial, const std::vector<cv::Point2f> &left, const std::vector<cv::Point2f> &right, const std::vector<int> &octave, float threshold, std::vector<bool> &inliers);
	cv::Matx33f estimateOpenCV(const cv::Matx33f &initial, const std::vector<cv::Point2f> &left, const std::vector<cv::Point2f> &right, const std::vector<int> &octave, float threshold, std::vector<bool> &inliers);

	cv::Matx33f estimateCeres(const cv::Matx33f &initial, const std::vector<cv::Point2f> &left, const std::vector<cv::Point2f> &right, const std::vector<int> &octave, const std::vector<double> &weights, float threshold, std::vector<bool> &inliers);

	////////////////////////////////////////////////
	// Estimation based on direct pixel comparison
	////////////////////////////////////////////////
    /**
     * @brief Estimates a similarity transform that minimizes the SSD between the aligned images.
     * This is the same method as used in PTAM.
     */
    bool estimateSimilarityDirect(const cv::Mat1b &imgRef, const cv::Mat1b &imgNew, cv::Matx23f &transform);
    bool estimateSimilarityDirect(const cv::Mat1b &imgRef, const cv::Mat1s &imgRefDx, const cv::Mat1s &imgRefDy, const cv::Mat1b &imgNew, cv::Matx23f &transform);


	////////////////////////////////////////////////
	// These versions are used for testing only (can be optimized a bit more)
	////////////////////////////////////////////////

    //This version accepts a homography as input
    //The calculated similarity is applied on top of the input transform
    bool estimateSimilarityDirect(const cv::Mat1b &imgRef, const cv::Mat1b &imgNew, cv::Matx33f &transform);

    bool estimateAffineDirect(const cv::Mat1b &imgRef, const cv::Mat1b &imgNew, cv::Matx33f &transform);

    bool estimateHomographyDirect(const cv::Mat1b &imgRef, const cv::Mat1b &imgNew, cv::Matx33f &transform);

private:
    int mMaxIterations;
    bool mShowIterations;
};

class HomographyReprojectionError
{
public:
	HomographyReprojectionError(double leftX, double leftY, double rightX, double rightY, double scale)
            : mLeftX(leftX), mLeftY(leftY), mRightX(rightX), mRightY(rightY), mScale(scale)
    {
    }

	//Homography is in row-major order
    template<typename T>
    bool operator()(const T * const homography, T *residuals) const
    {
        T p[3];

        //Translate
        p[0] = homography[0]*T(mRightX) + homography[1]*T(mRightY) + homography[2];
        p[1] = homography[3]*T(mRightX) + homography[4]*T(mRightY) + homography[5];
        p[2] = homography[6]*T(mRightX) + homography[7]*T(mRightY) + homography[8];

        //Normalize
        p[0] /= p[2];
        p[1] /= p[2];

        //Residuals
        residuals[0] = (T(mLeftX) - p[0]) / T(mScale);
        residuals[1] = (T(mLeftY) - p[1]) / T(mScale);
        return true;
    }

private:
    const double mLeftX;
    const double mLeftY;
    const double mRightX;
    const double mRightY;
    const double mScale;
};

}

#endif /* HOMOGRAPHYREPROJECTIONERROR_H_ */
