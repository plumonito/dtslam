/*
 * CeresParametrization.cpp
 *
 * Copyright(C) 2014, University of Oulu, all rights reserved.
 * Copyright(C) 2014, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
*/

#ifndef FIXED3DNORMPARAMETRIZATION_H_
#define FIXED3DNORMPARAMETRIZATION_H_

#include <ceres/local_parameterization.h>
#include "CeresUtils.h"

namespace dtslam
{

/**
 * @brief A parameterization class that is used for CERES solver. It parametrizes a 3D vector (like a translation) with two components, keeping the L2 norm fixed
 */
class Fixed3DNormParametrization: public ceres::LocalParameterization
{
public:
    Fixed3DNormParametrization(double norm)
            : mFixedNorm(norm)
    {
    }
    virtual ~Fixed3DNormParametrization()
    {
    }

    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const
    {
        return 3;
    }
    virtual int LocalSize() const
    {
        return 2;
    }

    /**
     * @brief Calculates two vectors that are orthogonal to X. It first picks a non-colinear point C then basis1=(X-C) x C and basis2=X x basis1
     */
    static void GetBasis(const double *x, double *basis1, double *basis2);

protected:
    const double mFixedNorm;
};

/**
 * @brief A parameterization class that is used for CERES solver. It parametrizes a 4D vector with three components, keeping the L2 norm fixed
 */
class Fixed4DNormParametrization: public ceres::LocalParameterization
{
public:
    Fixed4DNormParametrization(double norm)
            : mFixedNorm(norm)
    {
    }
    virtual ~Fixed4DNormParametrization()
    {
    }

    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const
    {
        return 4;
    }
    virtual int LocalSize() const
    {
        return 3;
    }

    /**
     * @brief Calculates 3 vectors that are orthogonal to X.
     */
    static void GetBasis(const double *x, double *basis1, double *basis2, double *basis3);

protected:
    const double mFixedNorm;
};

/*
 * I think specifying the full class (as above) is faster
struct Fixed4DNormPlus
{
	template<typename T>
	bool operator()(const T* x, const T* delta, T* x_plus_delta) const
	{
		double xd[4];
	    double basis1[4];
	    double basis2[4];
	    double basis3[4];

	    //To double
	    xd[0] = CeresUtils::ToDouble(x[0]);
	    xd[1] = CeresUtils::ToDouble(x[1]);
	    xd[2] = CeresUtils::ToDouble(x[2]);
	    xd[3] = CeresUtils::ToDouble(x[3]);

	    //Translation is constrained
	    Fixed4DNormParametrization::GetBasis(xd, basis1, basis2, basis3);

	    x_plus_delta[0] = x[0] + delta[0] * T(basis1[0]) + delta[1] * T(basis2[0]) + delta[2] * T(basis3[0]);
	    x_plus_delta[1] = x[1] + delta[0] * T(basis1[1]) + delta[1] * T(basis2[1]) + delta[2] * T(basis3[1]);
	    x_plus_delta[2] = x[2] + delta[0] * T(basis1[2]) + delta[1] * T(basis2[2]) + delta[2] * T(basis3[2]);
	    x_plus_delta[3] = x[3] + delta[0] * T(basis1[3]) + delta[1] * T(basis2[3]) + delta[2] * T(basis3[3]);

	    T norm = ceres::sqrt(
	            x_plus_delta[0] * x_plus_delta[0] + x_plus_delta[1] * x_plus_delta[1] + x_plus_delta[2] * x_plus_delta[2]
	            + x_plus_delta[3] * x_plus_delta[3]);
	    T factor = T(1) / norm;
	    x_plus_delta[0] *= factor;
	    x_plus_delta[1] *= factor;
	    x_plus_delta[2] *= factor;
	    x_plus_delta[3] *= factor;

	    return true;
	}
};
*/
}

#endif /* FIXED3DNORMPARAMETRIZATION_H_ */
