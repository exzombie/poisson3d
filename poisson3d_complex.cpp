#include <complex>
#include <vector>
#include <array>
#include <fftw3.h>
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include "vdt/exp.h"
#include "vdt/sincos.h"

using cplx = std::complex<double>;
using Mat = Eigen::Matrix<double, 3, 3>;
using Arr = std::vector<double>;

inline static cplx exp(cplx x)
{
    double r = vdt::fast_exp(x.real());
    double s, c;
    vdt::fast_sincos(x.imag(), s, c);
    return r * cplx(c, s);
}

template <typename T>
class Arr3D
{
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = ssize_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using shape_type = std::array<size_type, 3>;

    Arr3D(size_type i, size_type j, size_type k):
	shape_{{i, j, k}}, data_((T*)fftw_malloc(i*j*k*sizeof(T))) {}

    Arr3D(const shape_type& _shape):
	shape_(_shape), data_((T*)fftw_malloc(elements()*sizeof(T))) {}

    Arr3D(size_type i, size_type j, size_type k, T* data):
	shape_{{i, j, k}}, data_(data) {
        allocated = false;
    }

    ~Arr3D() {
	if (allocated)
	    fftw_free(data_);
    }

    const shape_type& shape() const {
	return shape_;
    }

    size_type elements() const {
	return shape_[0] * shape_[1] * shape_[2];
    }

    T* data() {
	return data_;
    }

    const T* data() const {
	return data_;
    }

    T& operator()(size_type i, size_type j, size_type k) {
	return data_[i*shape_[2]*shape_[1] + j*shape_[2] + k];
    }

    const T& operator()(size_type i, size_type j, size_type k) const {
	return data_[i*shape_[2]*shape_[1] + j*shape_[2] + k];
    }

    T& operator[](size_type i) {
	return data_[i];
    }

    const T& operator[](size_type i) const {
	return data_[i];
    }

    const T* begin() const {
	return data_;
    }

    T* begin() {
	return data_;
    }

    const T* end() const {
	return data_ + elements();
    }

    T* end() {
	return data_ + elements();
    }

private:
    const shape_type shape_;
    pointer data_;
    bool allocated = true;
};

using RealData = Arr3D<double>;
using CplxData = Arr3D<cplx>;

struct Poisson3dAux
{
    /* A function that returns the phase factor.  k and l are
       meta-indices choosing the axes.
       rpi = -2*pi*(u/Nu, v/Nv, w/Nw)
    */
    cplx Ekl(uint k, uint l, const double rpi[3])
    {
	cplx phs = 0;
	for (uint i = 0; i < 4; ++i) {
	    double sk = signums[2*i], sl = signums[2*i+1];
	    phs += sk * sl * exp(cplx(0, rpi[k]*sk + rpi[l]*sl));
	}
	return phs;
    }

    cplx Ekk(uint k, const double rpi[3])
    {
	double s, c;
	vdt::fast_sincos(rpi[k], s, c);
	return 2. * c - 2.;
    }

    // A function that maps reciprocal-space indices to solution factors.
    cplx P(uint u, uint v, uint w)
    {
	double rpi[3] = {frqu[u], frqv[v], frqw[w]};
	cplx s = 0;
	for (uint k = 0; k < 3; k++) {
	    s += M(k, k) * hr[k] * hr[k] * Ekk(k, rpi);
	    for (uint j = k+1; j < 3; j++) {
		s += 2 * M(j, k) * hr[j] * hr[k] * Ekl(k, j, rpi) / 4.;
	    }
	}
	return s;
    }

    static const double signums[8];
    double hr[3]; // 1 / h  where h is step size for each dimension
    Mat M; // Precomputed inverse squared basis matrix
    Arr frqu, frqv, frqw; // phase arguments (i.e. -2*pi/N) in fftshift order
};

const double Poisson3dAux::signums[8] = {
    1., 1., -1., -1., 1., -1., -1., 1.
};

static Arr fftfreq(uint N)
{
    Arr f(N);
    uint lim = N % 2 ? (N-1)/2 : N/2-1;
    uint i;
    for (i = 0; i <= lim; ++i) {
	f[i] = i;
    }
    int j = N % 2 ? -(int)lim : -(int)N/2;
    for (; i < N; ++i) {
	f[i] = j++;
    }
    return f;
}

static CplxData poisson3d(const RealData& rhs, const Mat& basis,
			  cplx* phase_factors = nullptr)
{
    Poisson3dAux A;
    A.M = basis.transpose().inverse();
    A.M = (A.M * A.M.transpose()).eval();
    for (uint i = 0; i < 3; ++i) {
	A.hr[i] = rhs.shape()[i];
    }
    A.frqu = fftfreq(rhs.shape()[0]);
    A.frqv = fftfreq(rhs.shape()[1]);
    A.frqw = fftfreq(rhs.shape()[2]);
    for (auto& x: A.frqu)
	x *= -2. * M_PI / rhs.shape()[0];
    for (auto& x: A.frqv)
	x *= -2. * M_PI / rhs.shape()[1];
    for (auto& x: A.frqw)
	x *= -2. * M_PI / rhs.shape()[2];

    CplxData xform(rhs.shape());

    fftw_plan planA =
	fftw_plan_dft_3d(xform.shape()[0], xform.shape()[1], xform.shape()[2],
			 reinterpret_cast<fftw_complex*>(xform.data()),
			 reinterpret_cast<fftw_complex*>(xform.data()),
			 FFTW_BACKWARD, FFTW_MEASURE);
    fftw_plan planB =
	fftw_plan_dft_3d(xform.shape()[0], xform.shape()[1], xform.shape()[2],
			 reinterpret_cast<fftw_complex*>(xform.data()),
			 reinterpret_cast<fftw_complex*>(xform.data()),
			 FFTW_FORWARD, FFTW_MEASURE);

    for (uint i = 0, end = xform.elements(); i < end; ++i) {
	xform[i] = rhs[i];
    }

    fftw_execute(planA);

    CplxData P(rhs.shape()[0], rhs.shape()[1], rhs.shape()[2], phase_factors);
    const cplx N = xform.elements();
    for (uint u = 0, endu = xform.shape()[0]; u < endu; ++u) {
	for (uint v = 0, endv = xform.shape()[1]; v < endv; ++v) {
	    for (uint w = 0, endw = xform.shape()[2]; w < endw; ++w) {
		cplx tmp = A.P(u, v, w);
		xform(u, v, w) /= N * tmp;
		if (phase_factors)
		    P(u, v, w) = tmp;
	    }
	}
    }
    // Zero-frequency component iz zeroed-out because it blows-up the
    // solution. E.g., for electric field calculation, this is
    // equivalent to making the unit cell neutral by adding background
    // compensating charge.
    xform[0] = 0;

    fftw_execute(planB);

    fftw_destroy_plan(planA);
    fftw_destroy_plan(planB);

    return xform;
}

extern "C" int poisson3d(uint32_t* shape, double* data, double* basis,
			 void* phase_factors)
{
    RealData rhs(shape[0], shape[1], shape[2], data);
    Mat B;
    for (uint j = 0; j < 3; ++j) {
	for (uint i = 0; i < 3; ++i) {
	    B(i, j) = basis[3*i + j];
	}
    }

    CplxData C = poisson3d(rhs, B, (cplx*)phase_factors);
    int largeImag = 0;
    double maxImag = 1e-13 * shape[0] * shape[1] * shape[2];
    for (uint i = 0, end = rhs.elements(); i < end; i++) {
	data[i] = C[i].real();
	if (std::abs(C[i].imag()) > maxImag) {
	    std::cerr << "Error in poisson3d: a large imaginary component ("
		      << std::abs(C[i].imag()) << " > "
		      << maxImag << ") was produced!" << std::endl;
	    largeImag = 1;
	    break;
	}
    }
    return largeImag;
}

extern "C" int poisson3d_precomputed(uint32_t* shape, double* data,
				     cplx* phase_factors)
{
    uint N = shape[0] * shape[1] * shape[2];
    cplx* xform = (cplx*)fftw_malloc(N * sizeof(cplx));

    fftw_plan planA =
	fftw_plan_dft_3d(shape[0], shape[1], shape[2],
			 reinterpret_cast<fftw_complex*>(xform),
			 reinterpret_cast<fftw_complex*>(xform),
			 FFTW_BACKWARD, FFTW_MEASURE);
    fftw_plan planB =
	fftw_plan_dft_3d(shape[0], shape[1], shape[2],
			 reinterpret_cast<fftw_complex*>(xform),
			 reinterpret_cast<fftw_complex*>(xform),
			 FFTW_FORWARD, FFTW_MEASURE);

    for (uint i = 0; i < N; ++i)
	xform[i] = data[i];

    const cplx n = (cplx)N;
    fftw_execute(planA);
    for (uint i = 1; i < N; ++i)
	xform[i] /= n * phase_factors[i];
    xform[0] = 0.0;
    fftw_execute(planB);

    fftw_destroy_plan(planA);
    fftw_destroy_plan(planB);

    int largeImag = 0;
    double maxImag = 1e-15 * shape[0] * shape[1] * shape[2];
    for (uint i = 0; i < N; i++) {
	data[i] = xform[i].real();
	if (std::abs(xform[i].imag()) > maxImag) {
	    std::cerr << "Error in poisson3d: a large imaginary component ("
		      << std::abs(xform[i].imag()) << " > "
		      << maxImag << ") was produced!" << std::endl;
	    largeImag = 1;
	    break;
	}
    }

    fftw_free(xform);
    return largeImag;
}

#ifdef TESTBUILD

int main()
{
    // A test case to run through valgrind.
    RealData::shape_type shape = {11, 12, 16};
    uint32_t shape32[] = {(uint32_t)shape[0], (uint32_t)shape[1], (uint32_t)shape[2]};
    RealData rho(shape);
    for (auto& x: rho)
	x = 0;

    double B[] = {28.32600,   0.00000,   0.00000,
		  -2.29635,  28.23277,   0.00000,
		  -2.29635,  -2.49071,  28.12268};
    poisson3d(shape32, rho.data(), B);
    return 0;
}

#endif
