/*
 * erfc.h
 * Inspired by the cephes math library (by Stephen L. Moshier
 * moshier@na-net.ornl.gov) as well as actual code.
 * The Cephes library can be found here:  http://www.netlib.org/cephes/
 * 
 *  Created on: Apr 10, 2014
 *      Author: Jure Varlec
 */

/* 
 * VDT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _VDT_ERFC_
#define _VDT_ERFC_

#include "vdtcore_common.h"
#include "exp.h"
#include <limits>

#include <iostream>

namespace vdt{

namespace details{
const double M = 128.;
const double MINV = .0078125;
const double MAXLOG = 7.09782712893383996732e2;

const double PX1erf = 2.46196981473530512524E-10;
const double PX2erf = 5.64189564831068821977E-1;
const double PX3erf = 7.46321056442269912687E0;
const double PX4erf = 4.86371970985681366614E1;
const double PX5erf = 1.96520832956077098242E2;
const double PX6erf = 5.26445194995477358631E2;
const double PX7erf = 9.34528527171957607540E2;
const double PX8erf = 1.02755188689515710272E3;
const double PX9erf = 5.57535335369399327526E2;

const double QX1erf = 1.32281951154744992508E1;
const double QX2erf = 8.67072140885989742329E1;
const double QX3erf = 3.54937778887819891062E2;
const double QX4erf = 9.75708501743205489753E2;
const double QX5erf = 1.82390916687909736289E3;
const double QX6erf = 2.24633760818710981792E3;
const double QX7erf = 1.65666309194161350182E3;
const double QX8erf = 5.57535340817727675546E2;

const double RX1erf = 5.64189583547755073984E-1;
const double RX2erf = 1.27536670759978104416E0;
const double RX3erf = 5.01905042251180477414E0;
const double RX4erf = 6.16021097993053585195E0;
const double RX5erf = 7.40974269950448939160E0;
const double RX6erf = 2.97886665372100240670E0;

const double SX1erf = 2.26052863220117276590E0;
const double SX2erf = 9.39603524938001434673E0;
const double SX3erf = 1.20489539808096656605E1;
const double SX4erf = 1.70814450747565897222E1;
const double SX5erf = 9.60896809063285878198E0;
const double SX6erf = 3.36907645100081516050E0;

const double TX1erf = 9.60497373987051638749E0;
const double TX2erf = 9.00260197203842689217E1;
const double TX3erf = 2.23200534594684319226E3;
const double TX4erf = 7.00332514112805075473E3;
const double TX5erf = 5.55923013010394962768E4;

const double UX1erf = 3.35617141647503099647E1;
const double UX2erf = 5.21357949780152679795E2;
const double UX3erf = 4.59432382970980127987E3;
const double UX4erf = 2.26290000613890934246E4;
const double UX5erf = 4.92673942608635921086E4;

}

// Calculates exp(x*x) or exp(-x*x), suppressing error amplification.
inline double fast_expx2(double initial_x, bool negative){
    const double x = negative ? -std::fabs(initial_x) : std::fabs(initial_x);
    const double m = details::MINV * details::fpfloor(details::M * x + 0.5);
    const double f = x - m;
    double u = m * m;
    double u1 = 2 * m * f + f * f;
    if (negative) {
        u = -u;
        u1 = -u1;
    }

    if (u + u1 > details::MAXLOG)
        return std::numeric_limits<double>::infinity();
    else
        return fast_exp(u) * fast_exp(u1);
}

inline double fast_erfc(double);

// Calculates erf(x).
inline double fast_erf(double initial_x){
    const double x = initial_x;
    if (fabs(x) > 1.0) {
        return 1.0 - fast_erfc(x);
    }
    const double z = x * x;

    double t = details::TX1erf;
    t *= z;
    t += details::TX2erf;
    t *= z;
    t += details::TX3erf;
    t *= z;
    t += details::TX4erf;
    t *= z;
    t += details::TX5erf;

    double u = z;
    u += details::UX1erf;
    u *= z;
    u += details::UX2erf;
    u *= z;
    u += details::UX3erf;
    u *= z;
    u += details::UX4erf;
    u *= z;
    u += details::UX5erf;

    return x * t / u;
}

// Calculates erfc(x).
inline double fast_erfc(double initial_x){
    const double x = std::fabs(initial_x);
    if (x < 1.0) {
        return 1.0 - fast_erf(initial_x);
    }
    if (x > 6.0) {
        // 1e-16 is precise enough for me; the original code had x > 26 here,
        // but in terms of MAXLOG.
        // The code for x > 8.0 below is commented out for this reason.
        return initial_x < 0.0 ? 2.0 : 0.0;
    }

    const double ez = fast_expx2(x, true);
    double p, q;
    //if (x < 8.0) {
        p = details::PX1erf;
        p *= x;
        p += details::PX2erf;
        p *= x;
        p += details::PX3erf;
        p *= x;
        p += details::PX4erf;
        p *= x;
        p += details::PX5erf;
        p *= x;
        p += details::PX6erf;
        p *= x;
        p += details::PX7erf;
        p *= x;
        p += details::PX8erf;
        p *= x;
        p += details::PX9erf;

        q = x;
        q += details::QX1erf;
        q *= x;
        q += details::QX2erf;
        q *= x;
        q += details::QX3erf;
        q *= x;
        q += details::QX4erf;
        q *= x;
        q += details::QX5erf;
        q *= x;
        q += details::QX6erf;
        q *= x;
        q += details::QX7erf;
        q *= x;
        q += details::QX8erf;
        /*
    } else {
        p = details::RX1erf;
        p *= x;
        p += details::RX2erf;
        p *= x;
        p += details::RX3erf;
        p *= x;
        p += details::RX4erf;
        p *= x;
        p += details::RX5erf;
        p *= x;
        p += details::RX6erf;

        q = x;
        q += details::SX1erf;
        q *= x;
        q += details::SX2erf;
        q *= x;
        q += details::SX3erf;
        q *= x;
        q += details::SX4erf;
        q *= x;
        q += details::SX5erf;
        q *= x;
        q += details::SX6erf;
    }
    */
    double y = ez * p / q;
    if (initial_x < 0.0)
        y = 2.0 - y;

    return y != 0 ? y :
           initial_x < 0.0 ? 2.0 : 0.0;
}

}

#endif
