#ifndef arrow_writer_hpp
#define arrow_writer_hpp

#include <arrow/api.h>
#include <arrow/io/interfaces.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <iostream>
#include <sstream>
#include <pybind11/pybind11.h>

#include "utils.hpp"
#include "types.hpp"

#define check_arrow(s) check_arrow_status((s), __FILE__, __LINE__)

namespace py = pybind11;

inline void check_arrow_status(arrow::Status status,
                               const char* filename,
                               const std::size_t line_number) {
    if (!ARROW_PREDICT_FALSE(!status.ok())) return;
    
    std::ostringstream stream;
    stream << "arrow error: " << status.ToString() << " file: " << filename << " line: " << line_number;
    std::cout << stream.str() << std::endl;
    throw std::runtime_error(stream.str());
}

class ArrowWriter : public Writer {
public:
    explicit ArrowWriter(const std::string& output_file_prefix, const Schema& schema, bool create_batch_id_file, int max_batch_size = -1);
    void add_record(int line_number, const Tuple& tuple) override;
    void add_tuple(int line_number, const py::tuple& py_tuple);
    void write_batch(const std::string& batch_id = "") override;
    void close(bool success = true) override;
    virtual ~ArrowWriter();
private:
    inline void add_value(int index, bool arg)
    { check_arrow(reinterpret_cast<arrow::BooleanBuilder*>(_arrays[index])->Append(arg)); }
    inline void add_value(int index, int32_t arg)
    { check_arrow(reinterpret_cast<arrow::Int32Builder*>(_arrays[index])->Append(arg)); }
    inline void add_value(int index, int64_t arg)
    { check_arrow(reinterpret_cast<arrow::Int64Builder*>(_arrays[index])->Append(arg)); }
    inline void add_value(int index, float arg)
    { check_arrow(reinterpret_cast<arrow::FloatBuilder*>(_arrays[index])->Append(arg)); }
    inline void add_value(int index, double arg)
    { check_arrow(reinterpret_cast<arrow::DoubleBuilder*>(_arrays[index])->Append(arg)); }
    inline void add_value(int index, const std::string& arg)
    { check_arrow(reinterpret_cast<arrow::StringBuilder*>(_arrays[index])->Append(arg)); }
    
    arrow::ArrayBuilder* get_array_builder(int i);
    
    std::string _output_file_prefix;
    std::shared_ptr<arrow::Schema> _schema;
    std::shared_ptr<arrow::Schema> _id_schema;
    bool _create_batch_id_file;
    int _max_batch_size;
    std::vector<std::string> _batch_ids;
    std::vector<int> _line_num;
    int _record_num;
    std::shared_ptr<arrow::ipc::RecordBatchWriter> _batch_writer;
    std::shared_ptr<arrow::io::OutputStream> _output_stream;
    std::vector<void*> _arrays;
    bool _closed;
};

/*inline std::shared_ptr<Writer> arrow_writer_creator(const std::string& output_file_prefix, const Schema& schema, bool create_batch_id_file, int batch_size) {
    std::shared_ptr<ArrowWriter> ptr(new ArrowWriter(output_file_prefix, schema, create_batch_id_file, batch_size));
    return std::dynamic_pointer_cast<Writer>(ptr);
}*/

#endif /* arrow_writer_hpp */
