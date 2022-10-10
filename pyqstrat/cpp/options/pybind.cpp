//#ifdef __clang__
//#pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Weverything"
//#endif
#include <pybind11/pybind11.h>
//#ifdef __clang__
//#pragma clang diagnostic pop
//#endif

#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/iostream.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std;

void init_pybind_options(py::module &); //Initialize the black scholes options module

PYBIND11_MODULE(pyqstrat_cpp, m) {
    init_pybind_options(m);
    m.attr("__name__") = "pyqstrat.pyqstrat_cpp";
    py::options options;
    options.disable_function_signatures();
}

