/*
 * HomographyReprojectionError.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#include "HomographyEstimation.h"

#include <memory>
#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "log.h"
#include "cvutils.h"

namespace dtslam {

cv::Matx33f HomographyEstimation::estimateCeres(const cv::Matx33f &initial, const std::vector<cv::Point2f> &left, const std::vector<cv::Point2f> &right, const std::vector<int> &octave, float threshold, std::vector<bool> &inliers)
{
	ceres::Problem problem;
	ceres::LossFunction *lossFunc = new ceres::CauchyLoss(threshold);

	cv::Matx33d h(initial);
	for(int i=0; i<(int)left.size(); i++)
	{
		problem.AddResidualBlock(
				new ceres::AutoDiffCostFunction<HomographyReprojectionError,2,9>(
						new HomographyReprojectionError(left[i].x, left[i].y, right[i].x, right[i].y, 1<<octave[i])),
				lossFunc, h.val);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;

	ceres::Solve(options, &problem, &summary);

	//DTSLAM_LOG << summary.BriefReport() << "\n";

	inliers.resize(left.size());
	for(int i=0; i<(int)left.size(); i++)
	{
		HomographyReprojectionError err(left[i].x, left[i].y, right[i].x, right[i].y, 1<<octave[i]);
		double residuals[2];

		err(h.val, residuals);
		if(fabs(residuals[0])<threshold && fabs(residuals[1])<threshold)
			inliers[i] = true;
		else
			inliers[i] = false;
	}

	return h;
}

//Estimate homography with weights for each residual
cv::Matx33f HomographyEstimation::estimateCeres(const cv::Matx33f &initial, const std::vector<cv::Point2f> &left, const std::vector<cv::Point2f> &right, const std::vector<int> &octave, const std::vector<double> &weights, float threshold, std::vector<bool> &inliers)
{
	ceres::Problem problem;
	std::unique_ptr<ceres::LossFunction> lossFunc(new ceres::CauchyLoss(threshold));

	cv::Matx33d h(initial);
	for(int i=0; i<(int)left.size(); i++)
	{
		problem.AddResidualBlock(
				new ceres::AutoDiffCostFunction<HomographyReprojectionError,2,9>(
						new HomographyReprojectionError(left[i].x, left[i].y, right[i].x, right[i].y, 1<<octave[i])),
				new ceres::ScaledLoss(lossFunc.get(),weights[i],ceres::DO_NOT_TAKE_OWNERSHIP), h.val);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;

	ceres::Solve(options, &problem, &summary);

	//DTSLAM_LOG << summary.BriefReport() << "\n";

	//Mark inliers
	inliers.resize(left.size());
	for(int i=0; i<(int)left.size(); i++)
	{
		HomographyReprojectionError err(left[i].x, left[i].y, right[i].x, right[i].y, 1<<octave[i]);
		double residuals[2];

		err(h.val, residuals);
		if(fabs(residuals[0])<threshold && fabs(residuals[1])<threshold)
			inliers[i] = true;
		else
			inliers[i] = false;
	}

	return h;
}

cv::Matx33f HomographyEstimation::estimateOpenCV(const cv::Matx33f &initial, const std::vector<cv::Point2f> &left, const std::vector<cv::Point2f> &right, const std::vector<int> &octave, float threshold, std::vector<bool> &inliers)
{
	std::vector<uchar> inliers2(left.size());
	cv::Mat homo;
	homo = cv::findHomography(right, left, inliers2, cv::RANSAC, threshold);
	if(homo.empty())
	{
		DTSLAM_LOG << "Homography estimation failed.\n";
		return initial;
	}
	else
	{
		for(int i=0; i<(int)left.size(); i++)
			inliers[i] = (inliers2[i]!=0);
		return homo;
	}
}

bool HomographyEstimation::estimateSimilarityDirect(const cv::Mat1b &imgRef, const cv::Mat1b &imgNew, cv::Matx23f &transform)
{
    //Precalculate image jacobian
    cv::Mat1s imgRefDx;
    cv::Mat1s imgRefDy;
    cvutils::CalculateDerivatives(imgRef, imgRefDx, imgRefDy);

    return estimateSimilarityDirect(imgRef, imgRefDx, imgRefDy, imgNew, transform);
}

bool HomographyEstimation::estimateSimilarityDirect(const cv::Mat1b &imgRef, const cv::Mat1s &imgRefDx, const cv::Mat1s &imgRefDy, const cv::Mat1b &imgNew, cv::Matx23f &transform)
{
	bool converged = false;

    cv::Vec2i center = cv::Vec2i(imgRef.cols >> 1, imgRef.rows >> 1);

//	cv::Matx33f WfromC = cv::Matx33f::eye();
//    WfromC(0, 2) = center[0];
//    WfromC(1, 2) = center[1];
//
//    cv::Matx33f WfromCInv = cv::Matx33f::eye();
//    WfromCInv(0, 2) = -center[0];
//    WfromCInv(1, 2) = -center[1];

    //Initial transform

    //Iterate
    cv::Mat1b imgWarped;
    cv::Mat1s dxImgWarped;
    cv::Mat1s dyImgWarped;

    for(int k = 0; k < mMaxIterations; k++)
    {
        //Warp
        cv::warpAffine(imgNew, imgWarped, transform, cv::Size(imgRef.cols, imgRef.rows), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
		cvutils::CalculateDerivatives(imgWarped, dxImgWarped, dyImgWarped);

        /*
        const std::string kWarpedWindow("Warped back to original");
        cv::Mat1b imgDebug;
        cv::addWeighted(imgRef,0.5,imgWarped,0.5,0.0,imgDebug);
        cv::imshow(kWarpedWindow, imgDebug);
        cv::waitKey(50);
        */

        //Accumulators
        cv::Vec3i Jaccum(0, 0, 0);
        cv::Vec6i JtJTriangle(0, 0, 0, 0, 0, 0);
        for(int v = 1; v < imgWarped.rows - 1; v++)
        {
            for(int u = 1; u < imgWarped.cols - 1; u++)
            {
                uchar value00 = imgWarped(v - 1, u - 1);
                uchar value01 = imgWarped(v - 1, u - 0);
                uchar value02 = imgWarped(v - 1, u + 1);
                uchar value10 = imgWarped(v + 0, u - 1);
                uchar value11 = imgWarped(v + 0, u - 0);
                uchar value12 = imgWarped(v + 0, u + 1);
                uchar value20 = imgWarped(v + 1, u - 1);
                uchar value21 = imgWarped(v + 1, u - 0);
                uchar value22 = imgWarped(v + 1, u + 1);

                if(!value00 || !value01 || !value02 || !value10 || !value11 || !value12 || !value20 || !value21 || !value22)
                {
                    continue;    //A zero value might be real but we assume that it is out of bounds after the warp
                }

                //Intensity jacobian
                const short dxImgRefUV = imgRefDx(v, u);
                const short dyImgRefUV = imgRefDy(v, u);
                const short dxImgWarpedUV = dxImgWarped(v, u);
                const short dyImgWarpedUV = dyImgWarped(v, u);

                const short dxMean = (dxImgRefUV+dxImgWarpedUV)>>1;
                const short dyMean = (dyImgRefUV+dyImgWarpedUV)>>1;

                const cv::Vec2i Ji = cv::Vec2i(dxMean,
                                         dyMean); //The third component is always zero so we don't include it
                const cv::Vec3i Jesm(Ji[0], Ji[1], +(v - center[1])*Ji[0] - (u - center[0])*Ji[1]);

                const short diff = (short)imgWarped(v, u) - (short)imgRef(v, u);

                //Jaccum = sum(J'*y)
                Jaccum += diff * Jesm;

                //Lower left triangle of J'*J
                JtJTriangle[0] += Jesm[0] * Jesm[0];
                JtJTriangle[1] += Jesm[1] * Jesm[0];
                JtJTriangle[2] += Jesm[1] * Jesm[1];
                JtJTriangle[3] += Jesm[2] * Jesm[0];
                JtJTriangle[4] += Jesm[2] * Jesm[1];
                JtJTriangle[5] += Jesm[2] * Jesm[2];
            }
        }

        //Check score
        //if(score > lastScore)
        //    LOG("Minimization error. Cost increased. k=%d, lastScore=%.3f, score=%.3f",k,lastScore,score);

        //Assemble JtJ
        cv::Matx33f JtJ;
        for(int v = 0, j = 0; j < 3; j++)
            for(int i = 0; i <= j; i++)
            {
				JtJ(j, i) = JtJ(i, j) = (float)JtJTriangle[v++];
            }

        //Solve
	    cv::Vec3f params(0, 0, 0); //Similarity params (offsetX,offsetY,angle)
        cv::solve(JtJ, (cv::Vec3f)Jaccum, params, cv::DECOMP_SVD);

        if(fabs(params[0]) < 0.5f && fabs(params[1]) < 0.5f && fabs(params[2]) < 0.001f)
        {
            converged = true;
            break;
        }

        //Update transform
        cv::Matx23f updateMat;
        updateMat(0, 0) = cos(params[2]);
        updateMat(0, 1) = -sin(params[2]);
        updateMat(1, 0) = sin(params[2]);
        updateMat(1, 1) = cos(params[2]);
        updateMat(0, 2) = -params[0];
        updateMat(1, 2) = -params[1];

        //transform = transform*WfromCInv
        updateMat(0,2) += -updateMat(0,0)*center[0] - updateMat(0,1)*center[1];
        updateMat(1,2) += -updateMat(1,0)*center[0] - updateMat(1,1)*center[1];

        //transform = WfromC*transform
        updateMat(0,2) += center[0];
        updateMat(1,2) += center[1];

		transform = cvutils::AffineAffine(transform,updateMat);
    }

    return converged;
}

bool HomographyEstimation::estimateSimilarityDirect(const cv::Mat1b &imgRef, const cv::Mat1b &imgNew, cv::Matx33f &transform)
{
    bool converged = false;
    cv::Matx33f hInitial = transform;

	cv::Vec2f center = cv::Vec2f((float)(imgRef.cols / 2), (float)(imgRef.rows / 2));
    cv::Matx33f WfromC = cv::Matx33f::eye();
    WfromC(0, 2) = center[0];
    WfromC(1, 2) = center[1];

    cv::Matx33f WfromCInv = cv::Matx33f::eye();
    WfromCInv(0, 2) = -center[0];
    WfromCInv(1, 2) = -center[1];

    //Initial transform
    cv::Vec3f params(0, 0, 0);

    //Precalculate image jacobian
    cv::Mat1f dxImgRef;
    cv::Mat1f dyImgRef;
    cv::Sobel(imgRef, dxImgRef, CV_32F, 1, 0, 3);
    cv::Sobel(imgRef, dyImgRef, CV_32F, 0, 1, 3);

    //Iterate
    cv::Mat1b imgWarped;
    cv::Mat1f dxImgWarped;
    cv::Mat1f dyImgWarped;

    //float lastScore;
    float score = std::numeric_limits<float>::infinity();

    for(int k = 0; k < mMaxIterations; k++)
    {
        //lastScore = score;
        score = 0;

        //Accumulators
        cv::Vec3f Jaccum(0, 0, 0);
        cv::Vec6f JtJTriangle(0, 0, 0, 0, 0, 0);

        //Build homography
        cv::Matx33f CtoC;
        CtoC(0, 0) = cos(params[2]);
        CtoC(0, 1) = -sin(params[2]);
        CtoC(1, 0) = sin(params[2]);
        CtoC(1, 1) = cos(params[2]);
        CtoC(0, 2) = -params[0];
        CtoC(1, 2) = -params[1];
        CtoC(2, 0) = CtoC(2, 1) = 0;
        CtoC(2, 2) = 1;

        transform =  hInitial *WfromC * CtoC * WfromCInv;

        //Warp
        cv::warpPerspective(imgNew, imgWarped, transform, cv::Size(imgRef.cols, imgRef.rows), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);

        if(mShowIterations)
        {
			const std::string kWarpedWindow("Warped back to original");
			cv::Mat1b imgDebug;
			cv::addWeighted(imgRef,0.5,imgWarped,0.5,0.0,imgDebug);
			cv::imshow(kWarpedWindow, imgDebug);
			cv::waitKey(50);
        }

        cv::Sobel(imgWarped, dxImgWarped, CV_32F, 1, 0, 3);
        cv::Sobel(imgWarped, dyImgWarped, CV_32F, 0, 1, 3);

        for(int v = 1; v < imgWarped.rows - 1; v++)
        {
            for(int u = 1; u < imgWarped.cols - 1; u++)
            {
                uchar value00 = imgWarped(v - 1, u - 1);
                uchar value01 = imgWarped(v - 1, u - 0);
                uchar value02 = imgWarped(v - 1, u + 1);
                uchar value10 = imgWarped(v + 0, u - 1);
                uchar value11 = imgWarped(v + 0, u - 0);
                uchar value12 = imgWarped(v + 0, u + 1);
                uchar value20 = imgWarped(v + 1, u - 1);
                uchar value21 = imgWarped(v + 1, u - 0);
                uchar value22 = imgWarped(v + 1, u + 1);

                if(!value00 || !value01 || !value02 || !value10 || !value11 || !value12 || !value20 || !value21 || !value22)
                {
                    continue;    //A zero value might be real but we assume that it is out of bounds after the warp
                }

                //Intensity jacobian
                float dxImgRefUV = dxImgRef(v, u) / 8.0f;
                float dyImgRefUV = dyImgRef(v, u) / 8.0f;
                float dxImgWarpedUV = dxImgWarped(v, u) / 8.0f;
                float dyImgWarpedUV = dyImgWarped(v, u) / 8.0f;
                cv::Vec3f Ji = cv::Vec3f(0.5f * (dxImgRefUV + dxImgWarpedUV),
                                         0.5f * (dyImgRefUV + dyImgWarpedUV),
                                         0);
                cv::Vec3f Jesm(Ji[0], Ji[1], +(v - center[1])*Ji[0] - (u - center[0])*Ji[1]);

                float diff = (float)(imgWarped(v, u) - imgRef(v, u));
                score += diff * diff;

                //Jaccum = sum(J'*y)
                Jaccum += diff * Jesm;

                //Lower left triangle of J'*J
                JtJTriangle[0] += Jesm[0] * Jesm[0];
                JtJTriangle[1] += Jesm[1] * Jesm[0];
                JtJTriangle[2] += Jesm[1] * Jesm[1];
                JtJTriangle[3] += Jesm[2] * Jesm[0];
                JtJTriangle[4] += Jesm[2] * Jesm[1];
                JtJTriangle[5] += Jesm[2] * Jesm[2];
            }
        }

        //Check score
        //LOG("k=%d, lastScore=%.3f, score=%.3f",k,lastScore,score);
        //cvutils::Log(S);
        //if(score > lastScore)
        //    LOG("Minimization error. Cost increased. k=%d, lastScore=%.3f, score=%.3f",k,lastScore,score);

        //Assemble JtJ
        cv::Matx33f JtJ;
        for(int v = 0, j = 0; j < 3; j++)
            for(int i = 0; i <= j; i++)
            {
                JtJ(j, i) = JtJ(i, j) = JtJTriangle[v++];
            }

        //Solve
        cv::Vec3f paramUpdate;
        cv::solve(JtJ, Jaccum, paramUpdate, cv::DECOMP_SVD);
        params[0] += paramUpdate[0];
        params[1] += paramUpdate[1];
        params[2] += paramUpdate[2];

        if(fabs(paramUpdate[0]) < 0.5f && fabs(paramUpdate[1]) < 0.5f && fabs(paramUpdate[2]) < 0.001f)
        {
            //LOG("Converged, k=%d", k);
            converged = true;
            break;
        }
        //cvutils::Log(paramUpdate);
    }

    //if(!converged)
    //    LOG("Not converged!");

    return converged;
}

bool HomographyEstimation::estimateAffineDirect(const cv::Mat1b &imgRef, const cv::Mat1b &imgNew, cv::Matx33f &transform)
{
    bool converged = false;

    //Initial transform
    cv::Vec6f params(transform(0,0), transform(0,1), transform(0,2), transform(1,0) ,transform(1,1), transform(1,2));

    //Precalculate image jacobian
    cv::Mat1f dxImgRef;
    cv::Mat1f dyImgRef;
    cv::Sobel(imgRef, dxImgRef, CV_32F, 1, 0, 3);
    cv::Sobel(imgRef, dyImgRef, CV_32F, 0, 1, 3);

    //Iterate
    cv::Mat1b imgWarped;
    cv::Mat1f dxImgWarped;
    cv::Mat1f dyImgWarped;

    float lastScore;
    float score = std::numeric_limits<float>::infinity();

    for(int k = 0; k < mMaxIterations; k++)
    {
        lastScore = score;
        score = 0;

        //Accumulators
        cv::Vec6f Jaccum(0, 0, 0, 0, 0, 0);
        cv::Vec<float, 21> JtJTriangle;
        for(int i=0; i<21; i++)
        	JtJTriangle[i] = 0;

        //Build homography
        cv::Matx33f CtoC;
        CtoC(0, 0) = params[0];
        CtoC(0, 1) = params[1];
        CtoC(0, 2) = params[2];
        CtoC(1, 0) = params[3];
        CtoC(1, 1) = params[4];
        CtoC(1, 2) = params[5];
        CtoC(2, 0) = CtoC(2, 1) = 0;
        CtoC(2, 2) = 1;

        transform = CtoC;

        //Warp
        cv::warpPerspective(imgNew, imgWarped, transform, cv::Size(imgRef.cols, imgRef.rows), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);

//        const std::string kWarpedWindow("Warped back to original");
//        cv::Mat1b imgDebug;
//        cv::addWeighted(imgRef,0.5,imgWarped,0.5,0.0,imgDebug);
//        cv::imshow(kWarpedWindow, imgDebug);
//        cv::waitKey(50);

        cv::Sobel(imgWarped, dxImgWarped, CV_32F, 1, 0, 3);
        cv::Sobel(imgWarped, dyImgWarped, CV_32F, 0, 1, 3);

        for(int v = 1; v < imgWarped.rows - 1; v++)
        {
            for(int u = 1; u < imgWarped.cols - 1; u++)
            {
                uchar value00 = imgWarped(v - 1, u - 1);
                uchar value01 = imgWarped(v - 1, u - 0);
                uchar value02 = imgWarped(v - 1, u + 1);
                uchar value10 = imgWarped(v + 0, u - 1);
                uchar value11 = imgWarped(v + 0, u - 0);
                uchar value12 = imgWarped(v + 0, u + 1);
                uchar value20 = imgWarped(v + 1, u - 1);
                uchar value21 = imgWarped(v + 1, u - 0);
                uchar value22 = imgWarped(v + 1, u + 1);

                if(!value00 || !value01 || !value02 || !value10 || !value11 || !value12 || !value20 || !value21 || !value22)
                {
                    continue;    //A zero value might be real but we assume that it is out of bounds after the warp
                }

                //dIntensity/dX jacobian
                float dxImgRefUV = dxImgRef(v, u) / 8.0f;
                float dyImgRefUV = dyImgRef(v, u) / 8.0f;
                float dxImgWarpedUV = dxImgWarped(v, u) / 8.0f;
                float dyImgWarpedUV = dyImgWarped(v, u) / 8.0f;
                cv::Vec3f Ji = cv::Vec3f(0.5f * (dxImgRefUV + dxImgWarpedUV),
                                         0.5f * (dyImgRefUV + dyImgWarpedUV),
                                         0);

                //dX/dAffine jacobian
                cv::Matx<float, 3, 9> Jw = cv::Matx<float, 3, 9>::zeros();
				Jw(0, 0) = (float)u;
				Jw(0, 1) = (float)v;
                Jw(0, 2) = 1;
				Jw(0, 6) = (float)(-u * u);
				Jw(0, 7) = (float)(-u * v);
				Jw(0, 8) = (float)(-u);
				Jw(1, 3) = (float)u;
				Jw(1, 4) = (float)v;
                Jw(1, 5) = 1;
				Jw(1, 6) = (float)(-v * u);
				Jw(1, 7) = (float)(-v * v);
				Jw(1, 8) = (float)(-v);

                //dAffine/dParams jacobian
                cv::Matx<float, 9, 6> Jg = cv::Matx<float, 9, 6>::zeros();
                Jg(0, 0) = -1;
                Jg(1, 1) = -1;
                Jg(2, 2) = -1;
                Jg(3, 3) = -1;
                Jg(4, 4) = -1;
                Jg(5, 5) = -1;

                cv::Vec6f Jesm = (cv::Mat)(Ji.t() * Jw * Jg).t();

                float diff = (float)(imgWarped(v, u) - imgRef(v, u));
                score += diff * diff;

                //Jaccum = sum(J'*y)
                Jaccum += diff * Jesm;

                //Lower left triangle of J'*J
                JtJTriangle[0] += Jesm[0] * Jesm[0];

                JtJTriangle[1] += Jesm[1] * Jesm[0];
                JtJTriangle[2] += Jesm[1] * Jesm[1];

                JtJTriangle[3] += Jesm[2] * Jesm[0];
                JtJTriangle[4] += Jesm[2] * Jesm[1];
                JtJTriangle[5] += Jesm[2] * Jesm[2];

                JtJTriangle[6] += Jesm[3] * Jesm[0];
                JtJTriangle[7] += Jesm[3] * Jesm[1];
                JtJTriangle[8] += Jesm[3] * Jesm[2];
                JtJTriangle[9] += Jesm[3] * Jesm[3];

                JtJTriangle[10] += Jesm[4] * Jesm[0];
                JtJTriangle[11] += Jesm[4] * Jesm[1];
                JtJTriangle[12] += Jesm[4] * Jesm[2];
                JtJTriangle[13] += Jesm[4] * Jesm[3];
                JtJTriangle[14] += Jesm[4] * Jesm[4];

                JtJTriangle[15] += Jesm[5] * Jesm[0];
                JtJTriangle[16] += Jesm[5] * Jesm[1];
                JtJTriangle[17] += Jesm[5] * Jesm[2];
                JtJTriangle[18] += Jesm[5] * Jesm[3];
                JtJTriangle[19] += Jesm[5] * Jesm[4];
                JtJTriangle[20] += Jesm[5] * Jesm[5];
            }
        }

        //Check score
        DTSLAM_LOG << "k="<< k << ", lastScore=" << lastScore << ", score=" << score << "\n";
        if(score > lastScore)
        	DTSLAM_LOG << "Minimization error. Cost increased. k="<< k << ", lastScore=" << lastScore << ", score=" << score << "\n";

        //Assemble JtJ
        cv::Matx<float,6,6> JtJ;
        for(int v = 0, j = 0; j < 6; j++)
            for(int i = 0; i <= j; i++)
            {
                JtJ(j, i) = JtJ(i, j) = JtJTriangle[v++];
            }

        //Solve
        cv::Vec6f paramUpdate;
        cv::solve(JtJ, Jaccum, paramUpdate, cv::DECOMP_SVD);
        params[0] += paramUpdate[0];
        params[1] += paramUpdate[1];
        params[2] += paramUpdate[2];
        params[3] += paramUpdate[3];
        params[4] += paramUpdate[4];
        params[5] += paramUpdate[5];

        if(fabs(paramUpdate[0]) < 0.5f && fabs(paramUpdate[1]) < 0.5f && fabs(paramUpdate[2]) < 0.001f
        		&& fabs(paramUpdate[3]) < 0.5f && fabs(paramUpdate[4]) < 0.5f && fabs(paramUpdate[5]) < 0.001f)
        {
            //LOG("Converged, k=%d", k);
            converged = true;
            break;
        }
        //cvutils::Log(paramUpdate);
    }

    //if(!converged)
    //    LOG("Not converged!");

    return converged;
}

bool HomographyEstimation::estimateHomographyDirect(const cv::Mat1b &imgRef, const cv::Mat1b &imgNew, cv::Matx33f &transform)
{
    bool converged = false;

    //Initial transform
    cv::Vec<float, 9> params;
    for(int i=0; i<9; i++)
    	params[i] = transform.val[i];
//        params[0] = transform(0,2);
//        params[1] = transform(1,2);
//        params[2] = transform(0,1);
//        params[3] = transform(1,0);
//        params[4] = transform(0,0);
//        params[5] = transform(2,2);
//        params[6] = transform(2,0);
//        params[7] = transform(2,1);

    //Precalculate image jacobian
    cv::Mat1f dxImgRef;
    cv::Mat1f dyImgRef;
    cv::Sobel(imgRef, dxImgRef, CV_32F, 1, 0, 3);
    cv::Sobel(imgRef, dyImgRef, CV_32F, 0, 1, 3);

    //Iterate
    cv::Mat1b imgWarped;
    cv::Mat1f dxImgWarped;
    cv::Mat1f dyImgWarped;

    float lastScore;
    float score = std::numeric_limits<float>::infinity();

    for(int k = 0; k < mMaxIterations; k++)
    {
        lastScore = score;
        score = 0;

        //Accumulators
        cv::Vec<float,9> Jaccum;
        for(int i=0; i<9; i++)
        	Jaccum[i]=0;
        cv::Vec<float, 45> JtJTriangle;
        for(int i=0; i<45; i++)
        	JtJTriangle[i] = 0;

        //Build homography
        cv::Matx33f CtoC;
//        CtoC(0, 0) = params[4];
//        CtoC(0, 1) = params[2];
//        CtoC(0, 2) = params[0];
//        CtoC(1, 0) = params[3];
//        CtoC(1, 1) = -params[4]-params[5];
//        CtoC(1, 2) = params[1];
//        CtoC(2, 0) = params[6];
//        CtoC(2, 1) = params[7];
//        CtoC(2, 2) = params[5];
        for(int i=0; i<9; i++)
        	CtoC.val[i] = params[i];

        transform = CtoC;

        //Warp
        cv::warpPerspective(imgNew, imgWarped, transform, cv::Size(imgRef.cols, imgRef.rows), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);

//        const std::string kWarpedWindow("Warped back to original");
//        cv::Mat1b imgDebug;
//        cv::addWeighted(imgRef,0.5,imgWarped,0.5,0.0,imgDebug);
//        cv::imshow(kWarpedWindow, imgDebug);
//        cv::waitKey(50);

        cv::Sobel(imgWarped, dxImgWarped, CV_32F, 1, 0, 3);
        cv::Sobel(imgWarped, dyImgWarped, CV_32F, 0, 1, 3);

        for(int v = 1; v < imgWarped.rows - 1; v++)
        {
            for(int u = 1; u < imgWarped.cols - 1; u++)
            {
                uchar value00 = imgWarped(v - 1, u - 1);
                uchar value01 = imgWarped(v - 1, u - 0);
                uchar value02 = imgWarped(v - 1, u + 1);
                uchar value10 = imgWarped(v + 0, u - 1);
                uchar value11 = imgWarped(v + 0, u - 0);
                uchar value12 = imgWarped(v + 0, u + 1);
                uchar value20 = imgWarped(v + 1, u - 1);
                uchar value21 = imgWarped(v + 1, u - 0);
                uchar value22 = imgWarped(v + 1, u + 1);

                if(!value00 || !value01 || !value02 || !value10 || !value11 || !value12 || !value20 || !value21 || !value22)
                {
                    continue;    //A zero value might be real but we assume that it is out of bounds after the warp
                }

                //dIntensity/dX jacobian
                float dxImgRefUV = dxImgRef(v, u) / 8.0f;
                float dyImgRefUV = dyImgRef(v, u) / 8.0f;
                float dxImgWarpedUV = dxImgWarped(v, u) / 8.0f;
                float dyImgWarpedUV = dyImgWarped(v, u) / 8.0f;
                cv::Vec3f Ji = cv::Vec3f(0.5f * (dxImgRefUV + dxImgWarpedUV),
                                         0.5f * (dyImgRefUV + dyImgWarpedUV),
                                         0);

                //dX/dAffine jacobian
                cv::Matx<float, 3, 9> Jw = cv::Matx<float, 3, 9>::zeros();
				Jw(0, 0) = (float)u;
				Jw(0, 1) = (float)v;
                Jw(0, 2) = 1;
				Jw(0, 6) = (float)(-u * u);
				Jw(0, 7) = (float)(-u * v);
				Jw(0, 8) = (float)(-u);
				Jw(1, 3) = (float)u;
				Jw(1, 4) = (float)v;
                Jw(1, 5) = 1;
				Jw(1, 6) = (float)(-v * u);
				Jw(1, 7) = (float)(-v * v);
				Jw(1, 8) = (float)(-v);

                //dAffine/dParams jacobian
                cv::Matx<float, 9, 9> Jg = cv::Matx<float, 9, 9>::zeros();
                for(int i=0; i<9; i++)
                	Jg(i,i) = -1;
//                Jg(0, 4) = -1;
//                Jg(1, 2) = -1;
//                Jg(2, 0) = -1;
//                Jg(3, 3) = -1;
//                Jg(4, 4) = 1; Jg(4, 5) = 1;
//                Jg(5, 1) = -1;
//                Jg(6, 6) = -1;
//                Jg(7, 7) = -1;
//                Jg(8, 5) = -1;

                cv::Vec<float,9> Jesm = (cv::Mat)(Ji.t() * Jw * Jg).t();

                float diff = (float)(imgWarped(v, u) - imgRef(v, u));
                score += diff * diff;

                //Jaccum = sum(J'*y)
                Jaccum += diff * Jesm;

                //Lower left triangle of J'*J
                JtJTriangle[0] += Jesm[0] * Jesm[0];

                JtJTriangle[1] += Jesm[1] * Jesm[0];
                JtJTriangle[2] += Jesm[1] * Jesm[1];

                JtJTriangle[3] += Jesm[2] * Jesm[0];
                JtJTriangle[4] += Jesm[2] * Jesm[1];
                JtJTriangle[5] += Jesm[2] * Jesm[2];

                JtJTriangle[6] += Jesm[3] * Jesm[0];
                JtJTriangle[7] += Jesm[3] * Jesm[1];
                JtJTriangle[8] += Jesm[3] * Jesm[2];
                JtJTriangle[9] += Jesm[3] * Jesm[3];

                JtJTriangle[10] += Jesm[4] * Jesm[0];
                JtJTriangle[11] += Jesm[4] * Jesm[1];
                JtJTriangle[12] += Jesm[4] * Jesm[2];
                JtJTriangle[13] += Jesm[4] * Jesm[3];
                JtJTriangle[14] += Jesm[4] * Jesm[4];

                JtJTriangle[15] += Jesm[5] * Jesm[0];
                JtJTriangle[16] += Jesm[5] * Jesm[1];
                JtJTriangle[17] += Jesm[5] * Jesm[2];
                JtJTriangle[18] += Jesm[5] * Jesm[3];
                JtJTriangle[19] += Jesm[5] * Jesm[4];
                JtJTriangle[20] += Jesm[5] * Jesm[5];

                JtJTriangle[21] += Jesm[6] * Jesm[0];
                JtJTriangle[22] += Jesm[6] * Jesm[1];
                JtJTriangle[23] += Jesm[6] * Jesm[2];
                JtJTriangle[24] += Jesm[6] * Jesm[3];
                JtJTriangle[25] += Jesm[6] * Jesm[4];
                JtJTriangle[26] += Jesm[6] * Jesm[5];
                JtJTriangle[27] += Jesm[6] * Jesm[6];

                JtJTriangle[28] += Jesm[7] * Jesm[0];
                JtJTriangle[29] += Jesm[7] * Jesm[1];
                JtJTriangle[30] += Jesm[7] * Jesm[2];
                JtJTriangle[31] += Jesm[7] * Jesm[3];
                JtJTriangle[32] += Jesm[7] * Jesm[4];
                JtJTriangle[33] += Jesm[7] * Jesm[5];
                JtJTriangle[34] += Jesm[7] * Jesm[6];
                JtJTriangle[35] += Jesm[7] * Jesm[7];

                JtJTriangle[36] += Jesm[8] * Jesm[0];
                JtJTriangle[37] += Jesm[8] * Jesm[1];
                JtJTriangle[38] += Jesm[8] * Jesm[2];
                JtJTriangle[39] += Jesm[8] * Jesm[3];
                JtJTriangle[40] += Jesm[8] * Jesm[4];
                JtJTriangle[41] += Jesm[8] * Jesm[5];
                JtJTriangle[42] += Jesm[8] * Jesm[6];
                JtJTriangle[43] += Jesm[8] * Jesm[7];
                JtJTriangle[44] += Jesm[8] * Jesm[8];
            }
        }

        //Check score
        DTSLAM_LOG << "k="<< k << ", lastScore=" << lastScore << ", score=" << score << "\n";
        if(score > lastScore)
        	DTSLAM_LOG << "Minimization error. Cost increased. k="<< k << ", lastScore=" << lastScore << ", score=" << score << "\n";

        //Assemble JtJ
        cv::Matx<float,9,9> JtJ;
        for(int v = 0, j = 0; j < 9; j++)
            for(int i = 0; i <= j; i++)
            {
                JtJ(j, i) = JtJ(i, j) = JtJTriangle[v++];
            }

        //Solve
        cv::Vec<float,9> paramUpdate;
        cv::solve(JtJ, Jaccum, paramUpdate, cv::DECOMP_SVD);
        params[0] += paramUpdate[0];
        params[1] += paramUpdate[1];
        params[2] += paramUpdate[2];
        params[3] += paramUpdate[3];
        params[4] += paramUpdate[4];
        params[5] += paramUpdate[5];
        params[6] += paramUpdate[6];
        params[7] += paramUpdate[7];
        params[8] += paramUpdate[8];

        if(fabs(paramUpdate[0]) < 0.5f && fabs(paramUpdate[1]) < 0.5f && fabs(paramUpdate[2]) < 0.001f
        		&& fabs(paramUpdate[3]) < 0.5f && fabs(paramUpdate[4]) < 0.5f && fabs(paramUpdate[5]) < 0.001f
        		&& fabs(paramUpdate[6]) < 0.5f && fabs(paramUpdate[7]) < 0.5f && fabs(paramUpdate[8]) < 0.5f)
        {
            //LOG("Converged, k=%d", k);
            converged = true;
            break;
        }
        //cvutils::Log(paramUpdate);
    }

    //if(!converged)
    //    LOG("Not converged!");

    return converged;
}

} /* namespace dtslam */
