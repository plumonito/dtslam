/*
 * FixedNormParametrization.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
*/

#include "CeresParametrization.h"
#include <opencv2/core/core.hpp>

namespace dtslam
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// Fixed3DNormParametrization
///////////////////////////////////////////////////////////////////////////////////////////////////

// Calculates two vectors that are orthogonal to X.
// It first picks a non-colinear point C then basis1=(X-C) x C and basis2=X x basis1
void Fixed3DNormParametrization::GetBasis(const double *x, double *basis1, double *basis2)
{
    const double kThreshold = 0.1;

    //Check that the point we use is not colinear with x
    if (x[1] > kThreshold || x[1] < -kThreshold || x[2] > kThreshold || x[2] < -kThreshold)
    {
        //Use C=[1,0,0]
        basis1[0] = 0;
        basis1[1] = x[2];
        basis1[2] = -x[1];

        basis2[0] = -(x[1] * x[1] + x[2] * x[2]);
        basis2[1] = x[0] * x[1];
        basis2[2] = x[0] * x[2];
    }
    else
    {
        //Use C=[0,1,0]
        basis1[0] = -x[2];
        basis1[1] = 0;
        basis1[2] = x[0];

        basis2[0] = x[0] * x[1];
        basis2[1] = -(x[0] * x[0] + x[2] * x[2]);
        basis2[2] = x[1] * x[2];
    }
    double norm;
    norm = sqrt(basis1[0] * basis1[0] + basis1[1] * basis1[1] + basis1[2] * basis1[2]);
    basis1[0] /= norm;
    basis1[1] /= norm;
    basis1[2] /= norm;

    norm = sqrt(basis2[0] * basis2[0] + basis2[1] * basis2[1] + basis2[2] * basis2[2]);
    basis2[0] /= norm;
    basis2[1] /= norm;
    basis2[2] /= norm;

//	cv::Matx31f xmat(x[0],x[1],x[2]);
//	cv::Matx33f umat;
//	cv::Matx<float,1,1> wmat;
//	cv::Matx<float,1,1> vtmat;
//	cv::SVDecomp(xmat, wmat, umat, vtmat, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
//
//	basis1[0] = umat(0,1);
//	basis1[1] = umat(1,1);
//	basis1[2] = umat(2,1);
//
//	basis2[0] = umat(0,2);
//	basis2[1] = umat(1,2);
//	basis2[2] = umat(2,2);
}

bool Fixed3DNormParametrization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    double basis1[3];
    double basis2[3];

    //Translation is constrained
    GetBasis(x, basis1, basis2);

    x_plus_delta[0] = x[0] + delta[0] * basis1[0] + delta[1] * basis2[0];
    x_plus_delta[1] = x[1] + delta[0] * basis1[1] + delta[1] * basis2[1];
    x_plus_delta[2] = x[2] + delta[0] * basis1[2] + delta[1] * basis2[2];

    double norm = sqrt(
            x_plus_delta[0] * x_plus_delta[0] + x_plus_delta[1] * x_plus_delta[1] + x_plus_delta[2] * x_plus_delta[2]);
    double factor = mFixedNorm / norm;
    x_plus_delta[0] *= factor;
    x_plus_delta[1] *= factor;
    x_plus_delta[2] *= factor;

    return true;
}

bool Fixed3DNormParametrization::ComputeJacobian(const double *x, double *jacobian) const
{
    cv::Matx32d &jacobian_ = *(cv::Matx32d *)jacobian;
    double basis1[3];
    double basis2[3];

    //Translation is special
    GetBasis(x, basis1, basis2);

    jacobian_(0, 0) = basis1[0];
    jacobian_(1, 0) = basis1[1];
    jacobian_(2, 0) = basis1[2];

    jacobian_(0, 1) = basis2[0];
    jacobian_(1, 1) = basis2[1];
    jacobian_(2, 1) = basis2[2];
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Fixed4DNormParametrization
///////////////////////////////////////////////////////////////////////////////////////////////////

// Calculates 3 vectors that are orthogonal to X.
void Fixed4DNormParametrization::GetBasis(const double *x, double *basis1, double *basis2, double *basis3)
{
	cv::Matx41f xmat((float)x[0], (float)x[1], (float)x[2], (float)x[3]);
	cv::Matx44f umat;
	cv::Matx<float,1,1> wmat;
	cv::Matx<float,1,1> vtmat;
	cv::SVDecomp(xmat, wmat, umat, vtmat, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	basis1[0] = umat(0,1);
	basis1[1] = umat(1,1);
	basis1[2] = umat(2,1);
	basis1[3] = umat(3,1);

	basis2[0] = umat(0,2);
	basis2[1] = umat(1,2);
	basis2[2] = umat(2,2);
	basis2[3] = umat(3,2);

	basis3[0] = umat(0,3);
	basis3[1] = umat(1,3);
	basis3[2] = umat(2,3);
	basis3[3] = umat(3,3);
}

bool Fixed4DNormParametrization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    double basis1[4];
    double basis2[4];
    double basis3[4];

    //Translation is constrained
    GetBasis(x, basis1, basis2, basis3);

    x_plus_delta[0] = x[0] + delta[0] * basis1[0] + delta[1] * basis2[0] + delta[2] * basis3[0];
    x_plus_delta[1] = x[1] + delta[0] * basis1[1] + delta[1] * basis2[1] + delta[2] * basis3[1];
    x_plus_delta[2] = x[2] + delta[0] * basis1[2] + delta[1] * basis2[2] + delta[2] * basis3[2];
    x_plus_delta[3] = x[3] + delta[0] * basis1[3] + delta[1] * basis2[3] + delta[2] * basis3[3];

    double norm = sqrt(
            x_plus_delta[0] * x_plus_delta[0] + x_plus_delta[1] * x_plus_delta[1] + x_plus_delta[2] * x_plus_delta[2]
            + x_plus_delta[3] * x_plus_delta[3]);
    double factor = mFixedNorm / norm;
    x_plus_delta[0] *= factor;
    x_plus_delta[1] *= factor;
    x_plus_delta[2] *= factor;
    x_plus_delta[3] *= factor;

    return true;
}

bool Fixed4DNormParametrization::ComputeJacobian(const double *x, double *jacobian) const
{
    cv::Matx43d &jacobian_ = *(cv::Matx43d *)jacobian;
    double basis1[4];
    double basis2[4];
    double basis3[4];

    //Translation is special
    GetBasis(x, basis1, basis2, basis3);

    jacobian_(0, 0) = basis1[0];
    jacobian_(1, 0) = basis1[1];
    jacobian_(2, 0) = basis1[2];
    jacobian_(3, 0) = basis1[3];

    jacobian_(0, 1) = basis2[0];
    jacobian_(1, 1) = basis2[1];
    jacobian_(2, 1) = basis2[2];
    jacobian_(3, 1) = basis2[3];

    jacobian_(0, 2) = basis3[0];
    jacobian_(1, 2) = basis3[1];
    jacobian_(2, 2) = basis3[2];
    jacobian_(3, 2) = basis3[3];
    return true;
}

}
