#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;
using namespace cellcloudx;

extern "C" {
    void fgt_model(double *x, double *w, double sigma, int p, int K, double e,
                   double *xc, double *A_k, int d, int Nx,
                   int *indxc, int *indx, int *xhead, int *xboxsz,
                   double *dist_C, double *C_k, int *heads, int *cinds,
                   double *dx, double *prods, int pd);

    void fgt_predict(double *y, double *xc, double *A_k, int Ny, double sigma, int K, double e, int d, int pd,
                     double *v, double *dx, double *prods, int *heads);
}

py::tuple _fgt_model(py::array_t<double> x,
                            py::array_t<double> w,
                            double sigma, int p, int K, double e) {
    auto x_buf = x.request();
    auto w_buf = w.request();

    if (x_buf.ndim != 2 || w_buf.ndim != 1) {
        throw std::invalid_argument("x must be 2D and w must be 1D");
    }

    int d = x_buf.shape[0];
    int Nx = x_buf.shape[1];
    int pd = 1; // pd = nchoosek(p + d - 1, d)

    std::vector<double> xc(d * K, 0);
    std::vector<double> A_k(pd * K, 0);
    std::vector<int> indxc(K, 0), indx(Nx, 0), xhead(K, 0), xboxsz(K, 0);
    std::vector<double> dist_C(Nx, 0), C_k(pd, 0), dx(d, 0), prods(pd, 0);
    std::vector<int> heads(d + 1, 0), cinds(pd, 0);

    fgt_model(static_cast<double *>(x_buf.ptr),
              static_cast<double *>(w_buf.ptr),
              sigma, p, K, e,
              xc.data(), A_k.data(),
              d, Nx,
              indxc.data(), indx.data(), xhead.data(), xboxsz.data(),
              dist_C.data(), C_k.data(), heads.data(), cinds.data(),
              dx.data(), prods.data(), pd);

    return py::make_tuple(py::array_t<double>({d, K}, xc.data()),
                          py::array_t<double>({pd, K}, A_k.data()));
}


py::array_t<double> _fgt_predict(py::array_t<double> y,
                                        py::array_t<double> xc,
                                        py::array_t<double> A_k,
                                        double sigma, double e) {
    auto y_buf = y.request();
    auto xc_buf = xc.request();
    auto A_k_buf = A_k.request();

    if (y_buf.ndim != 2 || xc_buf.ndim != 2 || A_k_buf.ndim != 2) {
        throw std::invalid_argument("y, xc, and A_k must all be 2D arrays");
    }

    int d = y_buf.shape[0];
    int Ny = y_buf.shape[1];
    int K = xc_buf.shape[1];
    int pd = A_k_buf.shape[0];

    std::vector<double> v(Ny, 0);
    std::vector<double> dx(d, 0), prods(pd, 0);
    std::vector<int> heads(d + 1, 0);

    fgt_predict(static_cast<double *>(y_buf.ptr),
                static_cast<double *>(xc_buf.ptr),
                static_cast<double *>(A_k_buf.ptr),
                Ny, sigma, K, e, d, pd,
                v.data(), dx.data(), prods.data(), heads.data());

    return py::array_t<double>({Ny}, v.data());
}

PYBIND11_MODULE(_fgt, m) {
    m.doc() = "Python bindings for Fast Gauss Transform";
    m.def("fgt_model", &_fgt_model, "Compute the FGT model");
    m.def("fgt_predict", &_fgt_predict, "Predict using FGT model");
}

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}