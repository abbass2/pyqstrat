#include <pybind11/pybind11.h>

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

