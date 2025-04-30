#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include "fgt.hpp"

namespace py = pybind11;
using namespace fgt;

PYBIND11_MODULE(FGT, m) {
    Eigen::initParallel();

    py::class_<Direct>(m, "Direct")
        .def(py::init<Matrix, double>())
        .def("compute", &Direct::compute);
        // .def("compute_impl", &Direct::compute_impl);

    // m.def("compute_impl", [](const Matrix& target, const Vector& weights ) {
    //     Direct direct_obj;
    //     return direct_obj.compute_impl(target, weights);
    // });

// #ifdef VERSION_INFO
//     m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
// #else
//     m.attr("__version__") = "dev";
// #endif
}



// py::class_<DirectTree>(m, "DirectTree")
//     .def(py::init(MatrixRef source, double bandwidth, double epsilon))
//     .def("compute", &Direct::compute)
//     .def("compute_impl", &Direct::compute_impl)

// py::class_<Ifgt>(m, "Ifgt")
//     .def(py::init( MatrixRef source, double bandwidth, double epsilon))
//     .def("compute", &Direct::compute)
//     .def("compute_impl", &Direct::compute_impl)


// // 封装其他功能函数，假设有独立的函数
// m.def("direct_function", [](const Matrix& input_matrix) {
//     Direct direct_obj;
//     return direct_obj.compute(input_matrix);  // 返回计算结果
// });

// m.def("direct_tree_function", [](const Matrix& input_data) {
//     DirectTree tree;
//     return tree.query(input_data);  // 返回查询结果
// });

// m.def("ifgt_function", [](const Matrix& source, const Matrix& target, const Vector& weights) {
//     Ifgt ifgt_obj(source, 1.0, 0.1);  // 假设初始化 Ifgt 对象时使用这些参数
//     return ifgt_obj.compute(target, weights);  // 调用 compute 函数
// });
