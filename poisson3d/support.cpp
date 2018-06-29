#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <complex>
#include <vector>
#include <array>
#include <fftw3.h>
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include "vdt/exp.h"
#include "vdt/sincos.h"

using complex = std::complex<double>;
using Mat = Eigen::Matrix<double, 3, 3>;
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
using CplxData = Arr3D<complex>;

struct Poisson3dAux
{
    /* A function that returns the phase factor.  k and l are
       meta-indices choosing the axes.
       rpi = -2*pi*(u/Nu, v/Nv, w/Nw)
    */
    double Ekl(uint k, uint l, const double rpi[3])
    {
	double c1, c2, s;
	vdt::fast_sincos(rpi[k] + rpi[l], s, c1);
	vdt::fast_sincos(rpi[k] - rpi[l], s, c2);
	return 2 * (c1 - c2);
    }

    double Ekk(uint k, const double rpi[3])
    {
	double s, c;
	vdt::fast_sincos(rpi[k], s, c);
	return 2. * c - 2.;
    }

    // A function that maps reciprocal-space indices to solution factors.
    double P(uint u, uint v, uint w)
    {
	double rpi[3] = {frqu[u], frqv[v], frqw[w]};
	double s = 0;
	for (uint k = 0; k < 3; k++) {
	    s += M(k, k) * hr[k] * hr[k] * Ekk(k, rpi);
	    for (uint j = k+1; j < 3; j++) {
		s += 2 * M(j, k) * hr[j] * hr[k] * Ekl(k, j, rpi) / 4.;
	    }
	}
	return s;
    }

    double hr[3]; // 1 / h  where h is step size for each dimension
    Mat M; // Precomputed inverse squared basis matrix
    Arr frqu, frqv, frqw; // phase arguments (i.e. -2*pi/N) in fftshift order
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

static RealData poisson3d(const RealData& rhs, const Mat& basis,
			  double* phase_factors = nullptr)
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

    RealData reald(rhs.shape());
    CplxData::shape_type cshape = rhs.shape();
    cshape[2] = cshape[2] / 2 + 1;
    CplxData xform(cshape);

    fftw_plan planA =
	fftw_plan_dft_r2c_3d(reald.shape()[0], reald.shape()[1], reald.shape()[2],
			     reinterpret_cast<double*>(reald.data()),
			     reinterpret_cast<fftw_complex*>(xform.data()),
			     FFTW_MEASURE);
    fftw_plan planB =
	fftw_plan_dft_c2r_3d(reald.shape()[0], reald.shape()[1], reald.shape()[2],
			     reinterpret_cast<fftw_complex*>(xform.data()),
			     reinterpret_cast<double*>(reald.data()),
			     FFTW_MEASURE);

    memcpy(reald.data(), rhs.data(), sizeof(double) * reald.elements());
    fftw_execute(planA);

    RealData P(cshape[0], cshape[1], cshape[2], phase_factors);
    const complex N = reald.elements();
    for (uint u = 0, endu = xform.shape()[0]; u < endu; ++u) {
	for (uint v = 0, endv = xform.shape()[1]; v < endv; ++v) {
	    for (uint w = 0, endw = xform.shape()[2]; w < endw; ++w) {
		double tmp = A.P(u, v, w);
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

    return reald;
}


// Python wrappers, they define a module named "support".

static PyObject* poisson3d_wrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *basis, *in, *phase;
    if (!PyArg_ParseTuple(args, "OOO", &basis, &in, &phase)) {
        return NULL;
    };

    Mat B;
    for (uint j = 0; j < 3; ++j) {
	for (uint i = 0; i < 3; ++i) {
          B(i, j) = *(double*)PyArray_GETPTR2(basis, i, j);
	}
    }

    npy_intp* shape = PyArray_DIMS(in);
    RealData rhs(shape[0], shape[1], shape[2], (double*)PyArray_DATA(in));
    RealData C = poisson3d(rhs, B, (double*)PyArray_DATA(phase));

    memcpy(PyArray_DATA(in), C.data(), sizeof(double) * rhs.elements());
    return Py_None;
}

static PyObject* poisson3d_precomputed(PyObject* self, PyObject* args)
{
    PyArrayObject *in, *phase;
    if (!PyArg_ParseTuple(args, "OO", &in, &phase)) {
        return NULL;
    };

    npy_intp* shape = PyArray_DIMS(in);
    uint N = shape[0] * shape[1] * shape[2];
    uint Nc = shape[0] * shape[1] * (shape[2] / 2 + 1);
    double*  reald = (double*)fftw_malloc(N * sizeof(double));
    complex* xform = (complex*)fftw_malloc(Nc * sizeof(complex));

    fftw_plan planA =
	fftw_plan_dft_r2c_3d(shape[0], shape[1], shape[2],
			     reinterpret_cast<double*>(reald),
			     reinterpret_cast<fftw_complex*>(xform),
			     FFTW_MEASURE);
    fftw_plan planB =
	fftw_plan_dft_c2r_3d(shape[0], shape[1], shape[2],
			     reinterpret_cast<fftw_complex*>(xform),
			     reinterpret_cast<double*>(reald),
			     FFTW_MEASURE);

    memcpy(reald, PyArray_DATA(in), sizeof(double) * N);

    const complex n = (complex)N;
    fftw_execute(planA);
    for (uint i = 1; i < Nc; ++i) {
        xform[i] /= n * ((double*)PyArray_DATA(phase))[i];
    }
    xform[0] = 0.0;
    fftw_execute(planB);

    fftw_destroy_plan(planA);
    fftw_destroy_plan(planB);

    memcpy(PyArray_DATA(in), reald, sizeof(double) * N);

    fftw_free(reald);
    fftw_free(xform);

    return Py_None;
}

static PyMethodDef methods[] = {
    {"poisson3d", poisson3d_wrapper, METH_VARARGS,
        "Wrapper for poisson3d solver."},
    {"poisson3d_precomputed", poisson3d_precomputed, METH_VARARGS,
        "Wrapper for poisson3d solver with precomputed factors."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "support",
                                    NULL, -1, methods};

PyMODINIT_FUNC PyInit_support()
{
    return PyModule_Create(&module);
}


#ifdef TESTBUILD

int main()
{
    // An exercise to run through valgrind.
    RealData::shape_type shape = {11, 12, 16};
    RealData::shape_type shapep = {11, 12, 16 / 2 + 1};
    uint32_t shape32[] = {(uint32_t)shape[0], (uint32_t)shape[1], (uint32_t)shape[2]};
    RealData rho(shape), rho2(shape);
    RealData phase(shapep);

    srand(time(0));
    for (uint i = 0; i < rho.elements(); ++i) {
	rho[i] = rho2[i] = rand();
    }

    double B[] = {28.32600,   0.00000,   0.00000,
		  -2.29635,  28.23277,   0.00000,
		  -2.29635,  -2.49071,  28.12268};
    poisson3d(shape32, rho.data(), B, phase.data());
    poisson3d_precomputed(shape32, rho2.data(), phase.data());
    for (uint i = 0; i < rho.elements(); ++i) {
	if (rho[i] != rho2[i])
	    return 1;
    }
    return 0;
}

#endif
