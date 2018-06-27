#include <complex>
#include <vector>
#include <array>
#include <fftw3.h>
#include <eigen3/Eigen/Eigen>
#include <iostream>

using cplx = std::complex<double>;
using Mat = Eigen::Matrix<double, 3, 3>;
using MatC = Eigen::Matrix<cplx, 3, 3>;
using Vec = Eigen::Vector3d;
using Arr = std::vector<double>;

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

// Map array index to the corresponding frequency.
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

static CplxData poisson3d(const RealData& rhs, const Mat& basis)
{
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

    Mat M = basis.transpose().inverse();
    M = (M * M.transpose()).eval();
    double V = basis.row(0).dot(basis.row(1).cross(basis.row(2)));

    Mat KK;
    for (uint row = 0; row < 3; ++row)
	KK.row(row) = 2 * M_PI *
	    basis.row((row+1)%3).cross(basis.row((row+2)%3)) / V;
    auto K = [&KK] (uint u, uint v, uint w) {
	return u*KK.row(0) + v*KK.row(1) + w*KK.row(2);
    };

    Mat a;
    for (uint row = 0; row < 3; ++row)
	a.row(row) = basis.row(row) / xform.shape()[row];

    const cplx elems = xform.elements();
    Vec N;
    for (uint i = 0; i < 3; ++i)
	N[i] = xform.shape()[i];
    const cplx I(0, 1);
    using std::exp;
    for (uint u = 0, endu = xform.shape()[0]; u < endu; ++u) {
	for (uint v = 0, endv = xform.shape()[1]; v < endv; ++v) {
	    for (uint w = 0, endw = xform.shape()[2]; w < endw; ++w) {
		Vec k = K(u, v, w);
		MatC P;
		for (uint i = 0; i < 3; ++i) {
		    cplx tmpi = I * a.row(i).dot(k);
		    P(i, i) = N[i] * N[i] * (exp(-tmpi) + exp(+tmpi) - 2.);
		    for (uint j = i+1; j < 3; ++j) {
			cplx tmpj = I * a.row(j).dot(k);
			P(i, j) = N[i] * N[j] / 4. *
			    (exp(-(tmpi + tmpj)) + exp(tmpi + tmpj)
			     - exp(-(tmpi - tmpj)) - exp(tmpi - tmpj));
		    }
		    for (uint j = 0; j < i; ++j) {
			cplx tmpj = I * a.row(j).dot(k);
			P(i, j) = N[i] * N[j] / 4. *
			    (exp(-(tmpi + tmpj)) + exp(tmpi + tmpj)
			     - exp(-(tmpi - tmpj)) - exp(tmpi - tmpj));
		    }
		}
		xform(u, v, w) /= elems * (M * P).trace();
	    }
	}
    }

    // Zero-frequency component is zeroed-out because it blows-up the
    // solution. E.g., for electric field calculation, this is
    // equivalent to making the unit cell neutral by adding background
    // compensating charge.
    xform[0] = 0;

    fftw_execute(planB);

    fftw_destroy_plan(planA);
    fftw_destroy_plan(planB);

    return xform;
}

extern "C" int poisson3d(uint32_t* shape, double* data, double* basis)
{
    RealData rhs(shape[0], shape[1], shape[2], data);
    Mat B;
    for (uint j = 0; j < 3; ++j) {
	for (uint i = 0; i < 3; ++i) {
	    B(i, j) = basis[3*i + j];
	}
    }
    CplxData C = poisson3d(rhs, B);
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
