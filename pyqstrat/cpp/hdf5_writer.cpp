
#include "hdf5_writer.hpp"
#include "aggregators.hpp"
#include <iostream>
#include <sstream>
#include <tuple>
#include <stdarg.h>


using namespace std;
using namespace H5;

static const hsize_t CHUNK_SIZE = 10000;

hid_t create_strtype() {
    hid_t str_type = H5Tcopy (H5T_C_S1);
    H5Tset_size (str_type, H5T_VARIABLE);
    H5Tset_cset(str_type, H5T_CSET_UTF8);
    return str_type;
}

static hid_t STR_TYPE = create_strtype();

void* create_array(Schema::Type type) {
    switch (type) {
        case Schema::Type::INT32:
            return new vector<int32_t>();
        case Schema::Type::INT64:
            return new vector<int64_t>();
        case Schema::Type::FLOAT32:
            return new vector<float>();
        case Schema::Type::FLOAT64:
            return new vector<double>();
        case Schema::Type::BOOL:
            return new vector<uint8_t>();
        case Schema::Type::STRING:
            return new vector<std::string>();
        case Schema::Type::TIMESTAMP_MILLI:
            return new vector<int64_t>();
        case Schema::Type::TIMESTAMP_MICRO:
            return new vector<int64_t>();
    }
    error("unknown type:" << type);
}

void delete_array(Schema::Type type, void* array) {
    switch (type) {
        case Schema::Type::INT32:
            delete reinterpret_cast<vector<int32_t>*>(array);
            break;
        case Schema::Type::INT64:
            delete reinterpret_cast<vector<int64_t>*>(array);
            break;
        case Schema::Type::FLOAT32:
            delete reinterpret_cast<vector<float>*>(array);
            break;
        case Schema::Type::FLOAT64:
            delete reinterpret_cast<vector<double>*>(array);
            break;
        case Schema::Type::BOOL:
            delete reinterpret_cast<vector<uint8_t>*>(array);
            break;
        case Schema::Type::STRING:
            delete reinterpret_cast<vector<string>*>(array);
            break;
        case Schema::Type::TIMESTAMP_MILLI:
            delete reinterpret_cast<vector<int64_t>*>(array);
            break;
        case Schema::Type::TIMESTAMP_MICRO:
            delete reinterpret_cast<vector<int64_t>*>(array);
            break;
        default:
            error("unknown type:" << type);
    }
}

DSetCreatPropList get_dataset_props(size_t max_size) {
    DSetCreatPropList ds_props;  // create dataset creation prop list
    if (max_size < CHUNK_SIZE) return ds_props;
    ds_props.setShuffle(); // Supposed to improve gzip and szip compression
    ds_props.setFletcher32();  // Add checksum for corruption
    hsize_t chunk_size[1]{CHUNK_SIZE};
    ds_props.setChunk(1, chunk_size);  // then modify it for compression
    ds_props.setDeflate(6); // Compression level set to 6
    return ds_props;
}

void write(const std::string& name, Group& group, vector<uint8_t>* vec) {
    hsize_t dims[1]{vec->size()};
    DataSpace space( 1, dims);
    DataSet dataset = group.createDataSet(name, PredType::NATIVE_UINT8, space, get_dataset_props(vec->size()));
    dataset.write(vec->data(), PredType::NATIVE_UINT8);
    vec->clear();
}

void write(const std::string& name, Group& group, vector<int32_t>* vec) {
    hsize_t dims[1]{vec->size()};
    DataSpace space( 1, dims);
    DataSet dataset = group.createDataSet(name, PredType::NATIVE_INT32, space, get_dataset_props(vec->size()));
    dataset.write(vec->data(), PredType::NATIVE_INT32);
    vec->clear();
}

void write(const std::string& name, Group& group, vector<int64_t>* vec) {
    hsize_t dims[1]{vec->size()};
    DataSpace space( 1, dims);
    DataSet dataset = group.createDataSet(name, PredType::NATIVE_INT64, space, get_dataset_props(vec->size()));
    dataset.write(vec->data(), PredType::NATIVE_INT64);
    vec->clear();
}

void write(const std::string& name, Group& group, vector<float>* vec) {
    hsize_t dims[1]{vec->size()};
    DataSpace space( 1, dims);
    DataSet dataset = group.createDataSet(name, PredType::NATIVE_FLOAT, space, get_dataset_props(vec->size()));
    dataset.write(vec->data(), PredType::NATIVE_FLOAT);
    vec->clear();
}

void write(const std::string& name, Group& group, vector<double>* vec) {
    hsize_t dims[1]{vec->size()};
    DataSpace space( 1, dims);
    DataSet dataset = group.createDataSet(name, PredType::NATIVE_DOUBLE, space, get_dataset_props(vec->size()));
    dataset.write(vec->data(), PredType::NATIVE_DOUBLE);
    vec->clear();
}

void write(const std::string& name, Group& group, vector<string>* vec) {
    hsize_t dims[1]{vec->size()};
    DataSpace space( 1, dims);
    DataSet dataset = group.createDataSet(name, STR_TYPE, space, get_dataset_props(vec->size()));
    vector<const char*> strvec = vector<const char*>(vec->size());
    for (size_t i = 0; i < vec->size(); ++i) {
        strvec[i] = (*vec)[i].c_str();
    }
    dataset.write(strvec.data(), STR_TYPE);
    vec->clear();
}

void write_data(Group& group, const std::pair<std::string, Schema::Type>& field, void* array) {
    if (field.second == Schema::Type::INT32) write(field.first, group, reinterpret_cast<vector<int32_t>*>(array));
    else if (field.second == Schema::Type::INT64) write(field.first, group, reinterpret_cast<vector<int64_t>*>(array));
    else if (field.second == Schema::Type::FLOAT32) write(field.first, group, reinterpret_cast<vector<float>*>(array));
    else if (field.second == Schema::Type::FLOAT64) write(field.first, group, reinterpret_cast<vector<double>*>(array));
    else if (field.second == Schema::Type::BOOL) write(field.first, group, reinterpret_cast<vector<uint8_t>*>(array));
    else if (field.second == Schema::Type::STRING) write(field.first, group, reinterpret_cast<vector<string>*>(array));
    else if (field.second == Schema::Type::TIMESTAMP_MILLI) write(field.first, group, reinterpret_cast<vector<int64_t>*>(array));
    else if (field.second == Schema::Type::TIMESTAMP_MICRO) write(field.first, group, reinterpret_cast<vector<int64_t>*>(array));
    else error("Unknown type: " << field.first << " " << field.second);
}

HDF5Writer::HDF5Writer(const std::string& output_file_prefix, const Schema& schema) :
    _output_file_prefix(output_file_prefix),
    _line_num(vector<int>()),
    _file(0),
    _schema(schema),
    _closed(true) {
    for (auto schema_record : schema.types) {
        _arrays.push_back(create_array(schema_record.second));
    }
    if (_output_file_prefix[_output_file_prefix.size() - 1] == '.')
        _output_file_prefix = _output_file_prefix.substr(0, _output_file_prefix.size() - 1);
        
    /*H5::FileAccPropList fileAccPropList = H5::FileAccPropList::DEFAULT;
    int    mdc_nelmts  = 4096; // h5: number of items in meta data cache
    size_t rdcc_nelmts = 4096; // h5: number of items in raw data chunk cache
    size_t rdcc_nbytes = 5 * 1024 * 1024; // h5: raw data chunk cache size (in bytes) per dataset
    double rdcc_w0     = 1.0;    // h5: preemption policy
    fileAccPropList.setCache(mdc_nelmts, rdcc_nelmts, rdcc_nbytes, rdcc_w0); */
    
    _file = shared_ptr<H5File>(new H5File(output_file_prefix + ".hdf5.tmp", H5F_ACC_TRUNC));
    _closed = false;
}
                               

void HDF5Writer::add_record(int line_number, const Tuple& tuple) {
    int i = 0;
    for (auto field : _schema.types) {
        Schema::Type id = field.second;
        switch (id) {
            case Schema::Type::BOOL:
                add_value(i, tuple.get<bool>(i));
                break;
            case Schema::Type::INT32:
                add_value(i, tuple.get<int32_t>(i));
                break;
            case Schema::Type::INT64:
                add_value(i, tuple.get<int64_t>(i));
                break;
            case Schema::Type::FLOAT32:
                add_value(i, tuple.get<float>(i));
                break;
            case Schema::Type::FLOAT64:
                add_value(i, tuple.get<double>(i));
                break;
            case Schema::Type::STRING:
                add_value(i, tuple.get<string>(i));
                break;
            case Schema::Type::TIMESTAMP_MILLI:
                add_value(i, tuple.get<int64_t>(i));
                break;
            case Schema::Type::TIMESTAMP_MICRO:
                add_value(i, tuple.get<int64_t>(i));
                break;
            default:
                error("unknown type" << id);
        }
        i++;
    }
    _line_num.push_back(line_number);
}

void HDF5Writer::add_tuple(int line_number, const py::tuple& tuple) {
    int32_t i32val;
    int64_t i64val;
    float fval;
    double dval;
    bool bval;
    string sval;
    
    int i = 0;
    for (auto field : _schema.types) {
        if (i == 0) { //skip line number
            i = 1;
            continue;
        }
        Schema::Type id = field.second;
        const py::object& val = tuple[i-1];
        switch (id) {
            case Schema::Type::INT32:
                i32val = val.cast<int32_t>();
                add_value(i, i32val);
                break;
            case Schema::Type::INT64:
                i64val = val.cast<int64_t>();
                add_value(i, i64val);
                break;
            case Schema::Type::FLOAT32:
                fval = val.cast<float>();
                add_value(i, fval);
                break;
            case Schema::Type::FLOAT64:
                dval = val.cast<double>();
                add_value(i, dval);
                break;
            case Schema::Type::BOOL:
                bval = val.cast<bool>();
                add_value(i, bval);
                break;
            case Schema::Type::STRING:
                sval = val.cast<string>();
                add_value(i, sval);
                break;
            default:
                error("unknown type" << id);
        }
        i++;
    }
    _line_num.push_back(line_number);
}

void HDF5Writer::write_batch(const std::string& batch_id) {
    if (batch_id.size() == 0) error("batch id was empty");
    
    if (_line_num.empty()) return;
    
    hsize_t dims[1];
    dims[0] = _line_num.size();
    DataSpace dspace(1, dims);
    
    std::string tmp_group_name = batch_id + ".tmp";
    if (_file->exists(tmp_group_name)) _file->unlink(tmp_group_name);
    
    Group group = _file->createGroup(tmp_group_name);
    
    DataSet dataset = group.createDataSet("line_num", PredType::NATIVE_INT, dspace, get_dataset_props(_line_num.size()));
    dataset.write(_line_num.data(), PredType::NATIVE_INT);
    _line_num.clear();
    
    int i = 0;
    for (auto field : _schema.types) {
        write_data(group, field, _arrays[i]);
        i++;
    }
    dataset.close();
    if (_file->exists(batch_id)) _file->unlink(batch_id);
    _file->move(tmp_group_name, batch_id);
    _batch_ids.push_back(batch_id);
}

void HDF5Writer::close(bool success) {
    if (_closed) return;
    
    if (!_batch_ids.empty()) {
        if (_file->exists("index.tmp")) _file->unlink("index.tmp");
        Group group = _file->createGroup("index.tmp");
        write("group_names", group, &_batch_ids);
        if (_file->exists("index")) _file->unlink("index");
        _file->move("index.tmp", "index");
    }
    _file->close();
    std::rename((_output_file_prefix + ".hdf5.tmp").c_str(), (_output_file_prefix + ".hdf5").c_str());
     _closed = true;
}

HDF5Writer::~HDF5Writer() {
    close();
    for (int i = 1; i < static_cast<int>(_arrays.size()); ++i) {
        delete_array(_schema.types[i].second, _arrays[i]);
    }
}

std::shared_ptr<Writer> HDF5WriterCreator::call(const std::string& output_file_prefix, const Schema& schema) {
    return shared_ptr<Writer>(new HDF5Writer(output_file_prefix, schema));
}

void test_hdf5_writer() {
    Schema schema;
    schema.types = vector<pair<string, Schema::Type>>{
        make_pair("a", Schema::BOOL),
        make_pair("b", Schema::INT32),
        make_pair("c", Schema::STRING),
        make_pair("d", Schema::FLOAT64)
    };
    auto writer = HDF5Writer("/tmp/test", schema);
    
    Tuple tuple;
    tuple.add(false);
    tuple.add(5);
    tuple.add(std::string("abc"));
    tuple.add(1.234);
    writer.add_record(1, tuple);
    
    Tuple tuple2;
    tuple2.add(true);
    tuple2.add(8);
    tuple2.add(std::string("de"));
    tuple2.add(4.567);
    writer.add_record(2, tuple2);
    
    writer.write_batch("hello");
    writer.close();
}

int main_hdfwriter() {
    test_hdf5_writer();
    cout << "Done" << endl;
    return 0;
}
