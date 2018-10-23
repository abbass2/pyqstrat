#include "arrow_writer.hpp"
#include "aggregators.hpp"
#include <iostream>
#include <sstream>
#include <tuple>
#include <stdarg.h>


using namespace std;
using arrow::io::FileOutputStream;
using arrow::ipc::RecordBatchWriter;
using arrow::ipc::RecordBatchFileWriter;

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}

shared_ptr<arrow::DataType> to_schema_type(Schema::Type type) {
    if (type == Schema::Type::INT32) return arrow::int32();
    if (type == Schema::Type::INT64) return arrow::int64();
    if (type == Schema::Type::FLOAT32) return arrow::float32();
    if (type == Schema::Type::FLOAT64) return arrow::float64();
    if (type == Schema::Type::STRING) return arrow::utf8();
    if (type == Schema::Type::BOOL) return arrow::boolean();
    if (type == Schema::Type::TIMESTAMP_MILLI) return arrow::timestamp(arrow::TimeUnit::MILLI);
    if (type == Schema::Type::TIMESTAMP_MICRO) return arrow::timestamp(arrow::TimeUnit::MICRO);
    error("unknown type:" << type);
}

void* get_builder(Schema::Type type) {
    if (type == Schema::Type::INT32) return new arrow::Int32Builder();
    if (type == Schema::Type::INT64 || type == Schema::Type::TIMESTAMP_MILLI || type == Schema::Type::TIMESTAMP_MICRO) return new arrow::Int64Builder();
    if (type == Schema::Type::FLOAT32) return new arrow::FloatBuilder();
    if (type == Schema::Type::FLOAT64) return new arrow::DoubleBuilder();
    if (type == Schema::Type::STRING) return new arrow::StringBuilder();
    if (type == Schema::Type::BOOL) return new arrow::BooleanBuilder();
    error("unknown type:" << type);
}

void open_arrow_file(const string& filename, shared_ptr<arrow::Schema> schema,
                     shared_ptr<arrow::ipc::RecordBatchWriter>& batch_writer,
                     shared_ptr<arrow::io::OutputStream>& output_stream) {
    check_arrow(FileOutputStream::Open(filename, &output_stream));
    check_arrow(RecordBatchFileWriter::Open(output_stream.get(), schema, &batch_writer));
}

ArrowWriter::ArrowWriter(const std::string& output_file_prefix, const Schema& schema, bool create_batch_id_file, int max_batch_size) :
    _output_file_prefix(output_file_prefix),
    _create_batch_id_file(create_batch_id_file),
    _max_batch_size(max_batch_size),
    _batch_ids(vector<string>()),
    _line_num(vector<int>()),
    _record_num(0),
    _batch_writer(0),
    _output_stream(0),
    _closed(true) {
        
    vector<shared_ptr<arrow::Field>> fields;
    fields.push_back(arrow::field("line_num", arrow::int32()));
    _arrays.push_back(get_builder(Schema::INT32));
    for (auto schema_record : schema.types) {
        fields.push_back(arrow::field(schema_record.first, to_schema_type(schema_record.second)));
        _arrays.push_back(get_builder(schema_record.second));
    }
    _schema = make_shared<arrow::Schema>(fields);
    if (_max_batch_size == -1) max_batch_size = 1000000;
    if (_output_file_prefix[_output_file_prefix.size() - 1] == '.')
        _output_file_prefix = _output_file_prefix.substr(0, _output_file_prefix.size() - 1);
    open_arrow_file(_output_file_prefix + ".arrow.tmp", _schema, _batch_writer, _output_stream);
    _closed = false;
}

void ArrowWriter::add_record(int line_number, const Tuple& tuple) {
    add_value(0, line_number);
    
    int i = 0;
    for (auto field : _schema->fields())
    {
        if (i == 0) { //skip line number
            i = 1;
            continue;
        }
        arrow::Type::type id = field->type()->id();
        switch (id) {
            case arrow::Type::INT32:
                add_value(i, tuple.get<int32_t>(i-1));
                break;
            case arrow::Type::INT64:
                add_value(i, tuple.get<int64_t>(i-1));
                break;
            case arrow::Type::FLOAT:
                add_value(i, tuple.get<float>(i-1));
                break;
            case arrow::Type::DOUBLE:
                add_value(i, tuple.get<double>(i-1));
                break;
            case arrow::Type::BOOL:
                add_value(i, tuple.get<bool>(i-1));
                break;
            case arrow::Type::STRING:
                add_value(i, tuple.get<string>(i-1));
                break;
            case arrow::Type::TIMESTAMP:
                add_value(i, tuple.get<int64_t>(i-1));
                break;
            default:
                error("unknown type" << id);
        }
        i++;
    }
    _line_num.push_back(line_number);
    _record_num ++;
    if (_record_num == _max_batch_size) {
        write_batch();
        _record_num = 0;
    }
}

void ArrowWriter::add_tuple(int line_number, const py::tuple& tuple) {
    int32_t i32val;
    int64_t i64val;
    float fval;
    double dval;
    bool bval;
    string sval;
    
    add_value(0, line_number);
    
    int i = 0;
    for (auto field : _schema->fields())
    {
        if (i == 0) { //skip line number
            i = 1;
            continue;
        }
        arrow::Type::type id = field->type()->id();
        const py::object& val = tuple[i-1];
        switch (id) {
            case arrow::Type::INT32:
                i32val = val.cast<int32_t>();
                add_value(i, i32val);
                break;
            case arrow::Type::INT64:
                i64val = val.cast<int64_t>();
                add_value(i, i64val);
                break;
            case arrow::Type::FLOAT:
                fval = val.cast<float>();
                add_value(i, fval);
                break;
            case arrow::Type::DOUBLE:
                dval = val.cast<double>();
                add_value(i, dval);
                break;
            case arrow::Type::BOOL:
                bval = val.cast<bool>();
                add_value(i, bval);
                break;
            case arrow::Type::STRING:
                sval = val.cast<string>();
                add_value(i, sval);
                break;
            default:
                error("unknown type" << id);
        }
        i++;
    }
    _line_num.push_back(line_number);
    _record_num++;
    if (_record_num == _max_batch_size) {
        write_batch();
        _record_num = 0;
    }
}

arrow::ArrayBuilder* ArrowWriter::get_array_builder(int i) {
    auto field = _schema->field(i);
    switch (field->type()->id()) {
        case arrow::Type::INT32:
            return reinterpret_cast<arrow::Int32Builder*>(_arrays[i]);
        case arrow::Type::INT64:
            return reinterpret_cast<arrow::Int64Builder*>(_arrays[i]);
        case arrow::Type::FLOAT:
            return reinterpret_cast<arrow::FloatBuilder*>(_arrays[i]);
        case arrow::Type::DOUBLE:
            return reinterpret_cast<arrow::DoubleBuilder*>(_arrays[i]);
        case arrow::Type::BOOL:
            return reinterpret_cast<arrow::BooleanBuilder*>(_arrays[i]);
        case arrow::Type::STRING:
            return reinterpret_cast<arrow::StringBuilder*>(_arrays[i]);
        case arrow::Type::TIMESTAMP:
            return reinterpret_cast<arrow::Int64Builder*>(_arrays[i]);
        default:
            error("unknown type" << field->type()->id());
    }
    return nullptr;
}

void ArrowWriter::write_batch(const std::string& batch_id) {
    if (_create_batch_id_file) {
        if (batch_id.size() == 0) error("batch id must be provided if create_batch_id_file was set");
        _batch_ids.push_back(batch_id);
    }
    vector<shared_ptr<arrow::Array>> arrays;
    for (int i = 0; i < static_cast<int>(_arrays.size()); ++i) {
        shared_ptr<arrow::Array> array;
        auto builder = get_array_builder(i);
        check_arrow(builder->Finish(&array));
        arrays.push_back(array);
    }
    size_t batch_size = arrays[0]->length();
    if (batch_size == 0) return;
    auto batch = arrow::RecordBatch::Make(_schema, batch_size, arrays);
    if (_record_num < _max_batch_size) batch = batch->Slice(0, _record_num);
    check_arrow(_batch_writer->WriteRecordBatch(*batch));
    _record_num = 0;
    
    for (int i = 0; i < static_cast<int>(_arrays.size()); ++i) {
        auto builder = get_array_builder(i);
        builder->Reset();
    }
}

void ArrowWriter::close(bool success) {
    if (_closed) return;
    if (_record_num > 0) {
        if (_create_batch_id_file) error("unsaved rows remaining: " << _record_num << " please call write_batch");
        write_batch();
    }
    if (_batch_writer) check_arrow(_batch_writer->Close());
    if (_output_stream) check_arrow(_output_stream->Close());
            
    if (_create_batch_id_file) {
        auto id_schema = make_shared<arrow::Schema>(vector<shared_ptr<arrow::Field>>{
            arrow::field("id", arrow::utf8()),
            arrow::field("batch_id", arrow::int32())
        });

        auto id_builder = shared_ptr<arrow::StringBuilder>(new arrow::StringBuilder());
        check_arrow(id_builder->AppendValues(_batch_ids));
        
        auto batch_id_builder = shared_ptr<arrow::Int32Builder>(new arrow::Int32Builder());
        auto vals = arange<int>(0, static_cast<int>(_batch_ids.size()));
        check_arrow(batch_id_builder->AppendValues(vals));
        
        std::shared_ptr<arrow::Array> id_array; check_arrow(id_builder->Finish(&id_array));
        std::shared_ptr<arrow::Array> batch_id_array; check_arrow(batch_id_builder->Finish(&batch_id_array));
        
        size_t id_batch_size = id_array->length();
        
        auto id_batch = arrow::RecordBatch::Make(id_schema, id_batch_size, {id_array, batch_id_array});
        std::shared_ptr<arrow::ipc::RecordBatchWriter> batch_id_writer;
        std::shared_ptr<arrow::io::OutputStream> batch_id_output_stream;
        open_arrow_file(_output_file_prefix + ".batch_ids.arrow.tmp", id_schema, batch_id_writer, batch_id_output_stream);
        check_arrow(batch_id_writer->WriteRecordBatch(*id_batch));
        check_arrow(batch_id_writer->Close());
        check_arrow(batch_id_output_stream->Close());
    }
        
    if (success) {
        std::rename((_output_file_prefix + ".arrow.tmp").c_str(), (_output_file_prefix + ".arrow").c_str());
        std::rename((_output_file_prefix + ".batch_ids.arrow.tmp").c_str(), (_output_file_prefix + ".batch_ids.arrow").c_str());
    }
    _closed = true;
 }

ArrowWriter::~ArrowWriter() {
    close();
    for (int i = 0; i < static_cast<int>(_arrays.size()); ++i) {
        auto builder = get_array_builder(i);
        delete builder;
    }
}

void test_arrow_writer() {
    
    Schema schema;
    schema.types = vector<pair<string, Schema::Type>>{make_pair("a", Schema::BOOL), make_pair("b", Schema::INT32)};
    auto writer = ArrowWriter("/tmp/test", schema, false, 1);
    
    Tuple tuple;
    tuple.add(true);
    tuple.add(5);
    writer.add_record(1, tuple);
    
    Tuple tuple2;
    tuple2.add(false);
    tuple2.add(8);
    writer.add_record(2, tuple2);

    Tuple tuple3;
    tuple3.add(true);
    tuple3.add(9);
    writer.add_record(3, tuple3);
    
    //writer.write_batch();
    writer.close();
}

std::shared_ptr<Writer> ArrowWriterCreator::call(const std::string& output_file_prefix, const Schema& schema, bool create_batch_id_file, int batch_size) {
    return shared_ptr<Writer>(new ArrowWriter(output_file_prefix, schema, create_batch_id_file, batch_size));
}
