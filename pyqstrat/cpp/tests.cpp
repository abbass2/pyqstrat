#include "py_import_call_execute.hpp"
#include "zip_reader.hpp"
#include "file_reader.hpp"
#include "hdf5_writer.hpp"
#include "text_file_parsers.hpp"
#include "test_quote_pair.hpp"
#include <cstdlib>


int run_python() {
    // Run python
    int argc = 4;
    const char *argv2[] = {
        "ipython",
        "/Users/sal/Developer/pyqstrat:/Users/sal/Developer",
        "pyqstrat.notebooks.processing_marketdata_files",
        //"obc.apps.research.spx_options.parse_algoseek_option_prices_minute",
        "run"};
    return import_call_execute(argc, argv2);
}

int main() {
    putenv(const_cast<char*>(std::string("HDF5_DEBUG=trace").c_str()));
    //test_zip_reader(); //TODO: Create test zip file
    //test_zip_file_reader();
    /*test_fixed_width_time_parser();
    test_hdf5_writer();
    test_hdf5_lib();
    test_quote_pair_processing();*/
    run_python();
    return 0;
}
