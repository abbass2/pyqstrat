
#include "hdf5_writer.hpp"
#include "aggregators.hpp"
#include <iostream>
#include <sstream>
#include <tuple>
#include <stdarg.h>


using namespace std;
using namespace H5;

typedef int64_t TIMESTAMP_TYPE;

static const hsize_t CHUNK_SIZE = 10000;

hid_t create_strtype() {
    hid_t str_type = H5Tcopy (H5T_C_S1);
    H5Tset_size (str_type, H5T_VARIABLE);
    H5Tset_cset(str_type, H5T_CSET_UTF8);
    return str_type;
}

const hid_t STR_TYPE = create_strtype();

bool path_exists(hid_t id, const string& path) {
    return H5Lexists( id, path.c_str(), H5P_DEFAULT ) > 0;
}

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
            return new vector<TIMESTAMP_TYPE>();
        case Schema::Type::TIMESTAMP_MICRO:
            return new vector<TIMESTAMP_TYPE>();
    }
    error("unknown type:" << type);
}

void clear_array(Schema::Type type, void* array) {
    switch (type) {
        case Schema::Type::INT32:
            reinterpret_cast<vector<int32_t>*>(array)->clear();
            break;
        case Schema::Type::INT64:
            reinterpret_cast<vector<int64_t>*>(array)->clear();
            break;
        case Schema::Type::FLOAT32:
            reinterpret_cast<vector<float>*>(array)->clear();
            break;
        case Schema::Type::FLOAT64:
            reinterpret_cast<vector<double>*>(array)->clear();
            break;
        case Schema::Type::BOOL:
            reinterpret_cast<vector<uint8_t>*>(array)->clear();
            break;
        case Schema::Type::STRING:
            reinterpret_cast<vector<string>*>(array)->clear();
            break;
        case Schema::Type::TIMESTAMP_MILLI:
            reinterpret_cast<vector<TIMESTAMP_TYPE>*>(array)->clear();
            break;
        case Schema::Type::TIMESTAMP_MICRO:
            reinterpret_cast<vector<TIMESTAMP_TYPE>*>(array)->clear();
            break;
        default:
            error("unknown type:" << type);
    }
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
            delete reinterpret_cast<vector<TIMESTAMP_TYPE>*>(array);
            break;
        case Schema::Type::TIMESTAMP_MICRO:
            delete reinterpret_cast<vector<TIMESTAMP_TYPE>*>(array);
            break;
        default:
            error("unknown type:" << type);
    }
}

H5::DataType map_schema_type(Schema::Type type) {
    switch(type) {
        case Schema::Type::FLOAT32:
            return PredType::NATIVE_FLOAT;
        case Schema::Type::INT32:
            return PredType::NATIVE_INT32;
        case Schema::Type::STRING:
            return STR_TYPE;
        case Schema::Type::BOOL:
            return PredType::NATIVE_UINT8;
        case Schema::Type::TIMESTAMP_MILLI:
            return PredType::NATIVE_INT64;
        case Schema::Type::TIMESTAMP_MICRO:
            return PredType::NATIVE_INT64;
        case Schema::Type::INT64:
            return PredType::NATIVE_INT64;
        case Schema::Type::FLOAT64:
            return PredType::NATIVE_DOUBLE;
        default:
            error("unknown type: " << type);
    }
}

void write_array(DataSet& dataset, Schema::Type type, size_t vec_len, const void* vec_data) {
    DataSpace *filespace = new DataSpace(dataset.getSpace());
    hsize_t dims_out[1];
    filespace->getSimpleExtentDims(dims_out, NULL);
    delete filespace;
    
    hsize_t inc_len[1]{vec_len};
    hsize_t curr_len = dims_out[0];
    hsize_t new_len[1]{curr_len + inc_len[0]};
    dataset.extend(new_len);
    
    // Select a hyperslab in extended portion of the dataset.
    hsize_t offset[1]{curr_len};
    filespace = new DataSpace(dataset.getSpace ());
    filespace->selectHyperslab(H5S_SELECT_SET, inc_len, offset);
    // Define memory space.
    DataSpace *memspace = new DataSpace(1, inc_len, NULL);
    
    // Write data to the extended portion of the dataset.
    dataset.write(vec_data, map_schema_type(type), *memspace, *filespace);
    delete filespace;
    delete memspace;
}

std::pair<size_t, void*> get_vec_props(void *array, Schema::Type type) {
    if (type == Schema::Type::FLOAT32) {
        auto v = reinterpret_cast<vector<float>*>(array);
        return make_pair(v->size(), v->data());
    }
    if (type == Schema::Type::INT32) {
        auto v = reinterpret_cast<vector<int32_t>*>(array);
        return make_pair(v->size(), v->data());
    }
    if (type == Schema::Type::STRING) {
        auto v = reinterpret_cast<vector<string>*>(array);
        return make_pair(v->size(), v->data());
    }
    if (type == Schema::Type::BOOL) {
        auto v = reinterpret_cast<vector<uint8_t>*>(array);
        return make_pair(v->size(), v->data());
    }
    if (type == Schema::Type::TIMESTAMP_MILLI) {
        auto v = reinterpret_cast<vector<TIMESTAMP_TYPE>*>(array);
        return make_pair(v->size(), v->data());
    }
    if (type == Schema::Type::TIMESTAMP_MICRO) {
        auto v = reinterpret_cast<vector<TIMESTAMP_TYPE>*>(array);
        return make_pair(v->size(), v->data());
    }
    if (type == Schema::Type::FLOAT64) {
        auto v = reinterpret_cast<vector<double>*>(array);
        return make_pair(v->size(), v->data());
    }
    if (type == Schema::Type::FLOAT32) {
        auto v = reinterpret_cast<vector<float>*>(array);
        return make_pair(v->size(), v->data());
    }
    error("unknown type: " << type);
}

void write_data(Group& group, const std::pair<std::string, Schema::Type>& field, void* array) {
    if (field.second == Schema::Type::STRING) {
        vector<string>* vec = reinterpret_cast<vector<string>*>(array);
        size_t len = vec->size();
        const char** strvec = new const char*[len];
        int i = 0;
        for (const string& str : *vec) {
            strvec[i] = str.c_str();
            i++;
        }
        DataSet dataset = group.openDataSet(field.first);
        write_array(dataset, field.second, vec->size(), static_cast<const void*>(strvec));
        delete[] strvec;
    } else {
        pair<size_t, const void*> vec_props = get_vec_props(array, field.second);
        DataSet dataset = group.openDataSet(field.first);
        write_array(dataset, field.second, vec_props.first, vec_props.second);
    }
    
    clear_array(field.second, array);
}

DSetCreatPropList get_dataset_props() {
    DSetCreatPropList ds_props;  // create dataset creation prop list
    ds_props.setShuffle(); // Supposed to improve gzip and szip compression
    ds_props.setFletcher32();  // Add checksum for corruption
    hsize_t chunk_size[1]{CHUNK_SIZE};
    ds_props.setChunk(1, chunk_size);  // then modify it for compression
    ds_props.setDeflate(6); // Compression level set to 6
    return ds_props;
}

void create_datasets(const Schema& schema, H5::Group group) {
    hsize_t dims[1]{0};
    hsize_t maxdims[1]{H5S_UNLIMITED};
    DataSpace space( 1, dims, maxdims);
    for (auto field : schema.types) {
        const std::string& name = field.first;
        Schema::Type type = field.second;
        if (path_exists(group.getId(), name)) continue;
        DataType dtype = map_schema_type(type);
        group.createDataSet(name, dtype, space, get_dataset_props());
    }
}

HDF5Writer::HDF5Writer(H5::Group group, const Schema& schema) :
    _group(group),
    _schema(schema),
    _record_num(0),
    _closed(true) {
    for (auto schema_record : schema.types) {
        _arrays.push_back(create_array(schema_record.second));
    }
    create_datasets(_schema, group);
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
                add_value(i, tuple.get<TIMESTAMP_TYPE>(i));
                break;
            case Schema::Type::TIMESTAMP_MICRO:
                add_value(i, tuple.get<TIMESTAMP_TYPE>(i));
                break;
            default:
                error("unknown type" << id);
        }
        i++;
    }
    _record_num++;
    if (_record_num == 100000) {
        flush();
        _record_num = 0;
    }
}

void HDF5Writer::add_pytuple(int line_number, const py::tuple& tuple) {
    int32_t i32val;
    int64_t i64val;
    float fval;
    double dval;
    bool bval;
    string sval;
    
    int i = 0;
    for (auto field : _schema.types) {
        Schema::Type id = field.second;
        const py::object& val = tuple[i];
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
}

void HDF5Writer::flush() {
    int i = 0;
    for (auto field : _schema.types) {
        write_data(_group, field, _arrays[i]);
        i++;
    }
}

void HDF5Writer::close(bool success) {
    if (_closed) return;
    flush();
    _group.close();
    for (int i = 1; i < static_cast<int>(_arrays.size()); ++i) {
        delete_array(_schema.types[i].second, _arrays[i]);
    }
    _closed = true;
}

HDF5Writer::~HDF5Writer() {
    close();
}

HDF5WriterCreator::HDF5WriterCreator(const std::string& output_file_prefix, const std::string& group_name_delimiter) :
_h5file(nullptr),
_output_file_prefix(output_file_prefix),
_group_name_delimiter(group_name_delimiter),
_closed(true) {
    if (_group_name_delimiter.size() > 1) error("delimiter size was more than 1: " << group_name_delimiter  << group_name_delimiter.size());
}

vector<string> split(const string& s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream token_stream(s);
    while (getline(token_stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

H5::Group find_or_create_group(H5::H5Location* parent, const string& group_name, const std::string& delimiter) {
    // Create or find a group.  Will create parent groups if needed if delimiter is specified
    if (delimiter.empty()) {
        if (path_exists(parent->getId(), group_name))
            return parent->openGroup(group_name);
        else
            return parent->createGroup(group_name);
    }
    
    istringstream iss(group_name);
    vector<string> subgroups = split(group_name, delimiter[0]);
    if (!subgroups.size()) error("empty group_name " << group_name);
    Group group;
    for (auto subgroup : subgroups) {
        if (path_exists(parent->getId(), subgroup))
            group = parent->openGroup(subgroup);
        else
            group = parent->createGroup(subgroup);
        parent = &group;
    }
    return group;
}

std::shared_ptr<Writer> HDF5WriterCreator::call(const std::string& quote_id, const Schema& schema) {
    if (!_h5file)  {
        string output_file_name = _output_file_prefix + ".hdf5.tmp";
        try {
            _h5file = shared_ptr<H5File>(new H5File(output_file_name, H5F_ACC_TRUNC));
        } catch(const H5::FileIException& ex) {
            error("Could not open file: " << output_file_name << " : " << ex.getDetailMsg() << endl);
        }
        _closed = false;
    }
    H5::Group grp = find_or_create_group(_h5file.get(), quote_id, _group_name_delimiter);
    
    auto p = _writers.find(quote_id);
    shared_ptr<Writer> writer;
    if (p == _writers.end()) {
        writer = shared_ptr<Writer>(new HDF5Writer(grp, schema));
        _writers.insert(make_pair(quote_id, writer));
    } else {
        writer = p->second;
    }
    return writer;
}


void HDF5WriterCreator::close() {
    if (_closed) return;
    for (auto p : _writers) {
        p.second->close();
    }
    _h5file->close();
    auto tmp_file_name = _output_file_prefix + ".hdf5.tmp";
    auto output_file_name = _output_file_prefix + ".hdf5";
    std::rename(tmp_file_name.c_str(), (output_file_name).c_str());
    _closed = true;
}

HDF5WriterCreator::~HDF5WriterCreator() {
    close();
}

Tuple create_test_tuple(bool a, int b, string c, double d) {
    Tuple tuple;
    tuple.add(a);
    tuple.add(b);
    tuple.add(c);
    tuple.add(d);
    return tuple;
}

void test_hdf5_writer() {
    Schema schema;
    schema.types = vector<pair<string, Schema::Type>>{
        make_pair("a", Schema::BOOL),
        make_pair("b", Schema::INT32),
        make_pair("c", Schema::STRING),
        make_pair("d", Schema::FLOAT64)
    };
    auto writer_creator = HDF5WriterCreator("/tmp/test_hdf5_writer", " ");
    auto writer1 = writer_creator.call("test grp1", schema);
    writer1->add_record(1, create_test_tuple(false, 5, "abc", 1.234));
    writer1->add_record(2, create_test_tuple(true, 6, "def", 4.567));
    auto writer2 = writer_creator.call("test grp2", schema);
    writer2->add_record(1, create_test_tuple(true, 11, "xy", 9.1));
    writer2->add_record(2, create_test_tuple(false, 16, "z", 9.2));
    auto writer3 = writer_creator.call("test grp1", schema);
    writer3->add_record(3, create_test_tuple(false, 7, "ghi", 5.1));
    writer3->add_record(2, create_test_tuple(true, 8, "jkl", 6.1));
    writer_creator.close();
}

void test_hdf5_lib() {
    
    DSetCreatPropList ds_props;  // create dataset creation prop list
    ds_props.setShuffle(); // Supposed to improve gzip and szip compression
    ds_props.setFletcher32();  // Add checksum for corruption
    hsize_t chunk_size[1]{10};
    ds_props.setChunk(1, chunk_size);  // then modify it for compression
    ds_props.setDeflate(6); // Compression level set to 6
    
    const char* FILE_NAME = "/tmp/test_hdf5_lib.hdf5";
    H5File file( FILE_NAME, H5F_ACC_TRUNC );
    
    int int_data[2]{0, 1};
    hsize_t dims[1];
    hsize_t maxdims[1]{H5S_UNLIMITED};
    dims[0] = 2;
    
    DataSpace fspace( 1, dims, maxdims );
    
    DataSet int_ds = file.createDataSet("test_int", PredType::NATIVE_INT, fspace, ds_props);
    int_ds.write(int_data, PredType::NATIVE_INT);
    int_ds.close();
    
    // write required size char array
    hid_t str_type = H5Tcopy (H5T_C_S1);
    H5Tset_size (str_type, H5T_VARIABLE);
    H5Tset_cset(str_type, H5T_CSET_UTF8);
    DataSpace fspace2( 1, dims, maxdims );
    vector<const char*> str_data {"abc", "def"};
    DataSet str_ds = file.createDataSet("test_str", str_type, fspace, ds_props);
    str_ds.write(str_data.data(), str_type);
    str_ds.close();
    
    DataSet ext_int_ds = file.openDataSet("test_int");
    DataSpace tmp_fspace(ext_int_ds.getSpace());
    hsize_t dims_out[1];
    tmp_fspace.getSimpleExtentDims(dims_out, NULL);
    
    int new_int[3]{2, 3, 4};
    hsize_t inc_len[1]{sizeof(new_int)/sizeof(new_int[0])};
    hsize_t curr_len = dims_out[0];
    hsize_t new_len[1]{curr_len + inc_len[0]};
    ext_int_ds.extend(new_len);
        
    // Select a hyperslab in extended portion of the dataset.
    hsize_t offset[1]{curr_len};
    DataSpace ext_fspace(ext_int_ds.getSpace());
    ext_fspace.selectHyperslab(H5S_SELECT_SET, inc_len, offset);
    // Define memory space.
    DataSpace memspace(1, inc_len, NULL);
    
    // Write data to the extended portion of the dataset.
    //ext_int_ds.write(new_int, PredType::NATIVE_INT, memspace, ext_fspace);
    ext_int_ds.write(new_int, PredType::NATIVE_INT, memspace, ext_fspace);
    ext_int_ds.close();
    file.close();
    
    // Reopen the file
    int        rdata[5];
    hsize_t    rdims[2];
    
    // Open the file and dataset.
    file.openFile(FILE_NAME, H5F_ACC_RDONLY);
    DataSet rds(file.openDataSet("test_int"));
    
    // Get the dataset's dataspace and creation property list.
    DataSpace rfs(rds.getSpace());
    
    // Get information to obtain memory dataspace.
    int rank = rfs.getSimpleExtentNdims();
    rfs.getSimpleExtentDims(rdims);
    
    DataSpace rms(rank, rdims, NULL);
    rds.read(rdata, PredType::NATIVE_INT, rms, rfs);
    
    cout << endl;
    for (size_t i = 0; i < rdims[0]; i++) {
        cout << " " << i << ":" <<  rdata[i];
    }
    cout << endl;
    // Close all objects and file.
    file.close();
    
}

