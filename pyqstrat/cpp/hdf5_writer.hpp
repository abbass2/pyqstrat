#ifndef hdf5_writer_hpp
#define hdf5_writer_hpp

#if _MSC_VER >= 1900
#undef timezone
#endif

#include <H5Cpp.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>

#include <pybind11/pybind11.h>

#include "utils.hpp"
#include "pq_types.hpp"

namespace py = pybind11;

class HDF5Writer : public Writer {
public:
    explicit HDF5Writer(H5::Group group, const Schema& schema);
    void add_record(int line_number, const Tuple& tuple) override;
    void add_pytuple(int line_number, const py::tuple& py_tuple);
    void close(bool success = true) override;
    void flush();
    virtual ~HDF5Writer();
private:
    void add_value(int index, bool arg)
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
    H5::Group _group;
    Schema _schema;
    std::vector<void*> _arrays;
    int _record_num;
    bool _closed;
};

struct HDF5WriterCreator : public WriterCreator {
    explicit HDF5WriterCreator(const std::string& output_file_prefix, const std::string& group_name_delimiter = "");
    std::shared_ptr<Writer> call(const std::string& quote_id, const Schema& schema) override;
    ~HDF5WriterCreator();
    void close();
 private:
    std::shared_ptr<H5::H5File> _h5file;
    std::unordered_map<std::string, std::shared_ptr<Writer>> _writers;
    std::string _output_file_prefix;
    std::string _group_name_delimiter;
    bool _closed;
};

void test_hdf5_lib();
void test_hdf5_writer();


#endif /* hdf5_writer_hpp */

