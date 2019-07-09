#ifndef hdf5_writer_hpp
#define hdf5_writer_hpp

#if _MSC_VER >= 1900
#undef timezone
#endif

#include <H5Cpp.h>

#include <iostream>
#include <sstream>
#include <vector>

#include <pybind11/pybind11.h>

#include "utils.hpp"
#include "pq_types.hpp"

namespace py = pybind11;

class HDF5Writer : public Writer {
public:
    explicit HDF5Writer(const std::string& output_file_prefix, const Schema& schema);
    void add_record(int line_number, const Tuple& tuple) override;
    void add_tuple(int line_number, const py::tuple& py_tuple);
    void write_batch(const std::string& batch_id) override;
    void close(bool success = true) override;
    virtual ~HDF5Writer();
private:
    inline void add_value(int index, bool arg)
    { reinterpret_cast<std::vector<uint8_t>*>(_arrays[index])->push_back(arg); }
    inline void add_value(int index, int32_t arg)
    { reinterpret_cast<std::vector<int32_t>*>(_arrays[index])->push_back(arg); }
    inline void add_value(int index, int64_t arg)
    { reinterpret_cast<std::vector<int64_t>*>(_arrays[index])->push_back(arg); }
    inline void add_value(int index, float arg)
    { reinterpret_cast<std::vector<float>*>(_arrays[index])->push_back(arg); }
    inline void add_value(int index, double arg)
    { reinterpret_cast<std::vector<double>*>(_arrays[index])->push_back(arg); }
    inline void add_value(int index, const std::string& arg)
    { reinterpret_cast<std::vector<std::string>*>(_arrays[index])->push_back(arg); }
    std::string _output_file_prefix;
    std::vector<int> _line_num;
    std::vector<std::string> _batch_ids;
    std::shared_ptr<H5::H5File> _file;
    Schema _schema;
    std::vector<void*> _arrays;
    bool _closed;
};

struct HDF5WriterCreator : public WriterCreator {
    std::shared_ptr<Writer> call(const std::string& output_file_prefix, const Schema& schema) override;
};

#endif /* hdf5_writer_hpp */

