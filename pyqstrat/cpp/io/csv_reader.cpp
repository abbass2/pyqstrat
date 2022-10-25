//
//  csv_reader.cpp
//  py_c_test
//
//  Created by Sal Abbasi on 9/12/22.
//

#include "csv_reader.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <zip.h>
#include <math.h>
#include <string.h>
#include <unordered_map>
#include "utils.hpp"


// Windows uses _strdup instead of non-standard strdup function
#ifdef _MSC_VER
    #define strdup _strdup
    #define _CRT_SECURE_NO_WARNINGS
    #include <BaseTsd.h>
    typedef SSIZE_T ssize_t;
#endif

using namespace std;

static const float NANF = nanf("");
static const double NAND = nan("");

string get_error(int err_num) {
    char errmsg[255];
#ifdef _MSC_VER
    ::strerror_s(errmsg, 255, err_num);
#else
    ::strerror_r(err_num, errmsg, 255);
#endif
    return string(errmsg);
}

vector<char*> tokenize_line(char *s, char delim, const vector<int>& col_indices) {
    vector<char*> ret;
    ret.reserve(16);
    size_t col_idx = 0;
    char* begin = s;
    size_t curr_col_idx = 0;
    size_t size = ::strlen(s);
    s[size] = delim;  // replace last \0 with delim so we can tokenize last column
    for (size_t i = 0; i < size + 1; ++i) {
        if (s[i] == delim) {
            s[i] = '\0';
            if (col_indices[col_idx] == static_cast<int>(curr_col_idx)) {
                ret.push_back(begin);
                col_idx += 1;
                if (col_idx == col_indices.size()) break;
            }
            begin = s + i + 1;
            curr_col_idx += 1;
        }
    }
    return ret;
}

float str_to_float(const char* str, char decimal_point, char thousands_separator) {
    // convert a string to a float
    float result = 0;
    bool zero = false;
    float sign = *str == '-' ? static_cast<void>(str++), -1.0f : 1.0f;
    if (*str == '0') zero = true;
    while ((*str >= '0' && *str <= '9') || (*str == thousands_separator)) {
        if (*str == thousands_separator) {
            str++;
            continue;
        }
        result *= 10;
        result += *str - '0';
        str++;
    }
    if (!zero && (result == 0)) return NANF;

    float multiplier = 0.1f;
    if (*str == decimal_point) {
        str++;
        while (*str >= '0' && *str <= '9') {
            result += (*str - '0') * multiplier;
            multiplier /= 10;
            str++;
        }
    }
    
    float power = 0.0f;
    result *= sign;
    if (*str == 'e' || *str == 'E') {
        str++;
        float powerer = *str == '-'? static_cast<void>(str++), 0.1f : 10.0f;
        
        while ((*str >= '0') && (*str <= '9')) {
            power *= 10;
            power += *str - '0';
            str++;
        }
        result *= pow(powerer, power);
    }
    return result;
}


double str_to_double(const char* str, char decimal_point, char thousands_separator) {
    // convert a string to a float
    double result = 0;
    bool zero = false;
    float sign = *str == '-' ? static_cast<void>(str++), -1.0f : 1.0f;
    if (*str == '0') zero = true;
    while ((*str >= '0' && *str <= '9') || (*str == thousands_separator)) {
        if (*str == thousands_separator) {
            str++;
            continue;
        }
        result *= 10;
        result += *str - '0';
        str++;
    }
    if (!zero && (result == 0)) return NAND;
    
    float multiplier = 0.1f;
    if (*str == decimal_point) {
        str++;
        while (*str >= '0' && *str <= '9') {
            result += (*str - '0') * multiplier;
            multiplier /= 10;
            str++;
        }
    }
    
    float power = 0.0f;
    result *= sign;
    if (*str == 'e' || *str == 'E') {
        str++;
        float powerer = *str == '-'? static_cast<void>(str++), 0.1f : 10.0f;
        
        while ((*str >= '0') && (*str <= '9')) {
            power *= 10;
            power += *str - '0';
            str++;
        }
        result *= pow(powerer, power);
    }
    return result;
}

int32_t str_to_int32(const char* str, char thousands_separator) {
    // convert a string to a int
    int result = 0;
    int sign = *str == '-' ? static_cast<void>(str++), -1 : 1;
    while ((*str >= '0' && *str <= '9') || (*str == thousands_separator)) {
        if (*str == thousands_separator) {
            str++;
            continue;
        }
        result *= 10;
        result += *str - '0';
        str++;
    }
    result *= sign;
    return result;
}

int64_t str_to_int64(const char* str, char thousands_separator) {
    // convert a string to a int
    int64_t result = 0;
    int sign = *str == '-' ? static_cast<void>(str++), -1 : 1;
    while ((*str >= '0' && *str <= '9') || (*str == thousands_separator)) {
        if (*str == thousands_separator) {
            str++;
            continue;
        }
        result *= 10;
        result += *str - '0';
        str++;
    }
    result *= sign;
    return result;
}

int8_t str_to_int8(const char* str) {
    // convert a string to a int
    auto len = strlen(str);
    if (len == 0) return 0;
    if (len == 4) {
        if (strcmp(str, "true") == 0) return 1;
        if (strcmp(str, "TRUE") == 0) return 1;
        if (strcmp(str, "True") == 0) return 1;
    }
    if (len == 5) {
        if (strcmp(str, "false") == 0) return 0;
        if (strcmp(str, "FALSE") == 0) return 0;
        if (strcmp(str, "False") == 0) return 0;
    }
    int8_t result = 0;
    int sign = *str == '-' ? static_cast<void>(str++), -1 : 1;
    while (*str >= '0' && *str <= '9') {
        result *= 10;
        result += *str - '0';
        str++;
    }
    result *= sign;
    return result;
}

template<typename T> T parse_string(const char* str) {
    return std::string(str);
}

template<> int32_t parse_string<int32_t>(const char* str) {
    return str_to_int32(str, ',');
}

template<> int64_t parse_string<int64_t>(const char* str) {
    return str_to_int64(str, ',');
}

template<> float parse_string<float>(const char* str) {
    return str_to_float(str, '.', ',');
}

template<> double parse_string<double>(const char* str) {
    return str_to_double(str, '.', ',');
}
  
template<> int8_t parse_string<int8_t>(const char* str) {
    return str_to_int8(str);
}

template<typename T> void add_value(const char* str, void* column) {
    T elem = parse_string<T>(str);
    auto vec = static_cast<vector<T>*>(column);
    vec->push_back(elem);
}

void add_line(const vector<char*>& fields, const vector<string>& dtypes, vector<void*>& data) {
    for (size_t i=0; i < dtypes.size(); ++i) {
        if (dtypes[i] == "f4") {
            add_value<float>(fields[i], data[i]);
        } else if (dtypes[i] == "f8") {
            add_value<double>(fields[i], data[i]);
        } else if (dtypes[i] == "i1") {
            add_value<int8_t>(fields[i], data[i]);
        } else if (dtypes[i] == "i4") {
            add_value<int32_t>(fields[i], data[i]);
        } else if (dtypes[i] == "i8") {
            add_value<int64_t>(fields[i], data[i]);
        } else if (dtypes[i].substr(0, 3) == "M8[") {
            add_value<int64_t>(fields[i], data[i]);
        } else if (dtypes[i][0] == 'S') {
            add_value<string>(fields[i], data[i]);
        } else {
            error("invalid type: " << dtypes[i] << " expected i1, i4, i8, f4, f8, M8[*] or S[n]");
        }
    }
}

template<typename T> vector<T>* create_vec(size_t max_rows) {
    auto vec = new vector<T>();
    vec->reserve(max_rows);
    return vec;
}

void* create_vector(const std::string& dtype, size_t max_rows) {
    if (dtype == "f4") {
        return create_vec<float>(max_rows);
    } else if (dtype == "f8") {
        return create_vec<double>(max_rows);
    } else if (dtype == "i1") {
        return create_vec<int8_t>(max_rows);
    } else if (dtype == "i4") {
        return create_vec<int32_t>(max_rows);
    } else if (dtype == "i8") {
        return create_vec<int64_t>(max_rows);
    } else if (dtype.substr(0, 3) == "M8[") {
        return create_vec<int64_t>(max_rows);
    } else if (dtype[0] == 'S') {
        return create_vec<string>(max_rows);
    } else {
        error("invalid type: " << dtype << " expected i1, i4, i8, f4, f8, M8[*] or S[n]");
    }
}

class ZipArchive {
public:
    static ZipArchive& get_instance() {
        static ZipArchive  instance; // Guaranteed to be destroyed.
        return instance;
    }
    ZipArchive(const ZipArchive&) = delete;
    void operator=(const ZipArchive&) = delete;
    
    ~ZipArchive() {
        if (_zip_archives.size()) {
            for (auto zip_archive: _zip_archives) {
                zip_close(zip_archive.second);
            }
        }
    }
    
    zip_t* get_archive(const std::string& zip_filename) {
        zip_t* zip_archive = nullptr;
        {
            std::lock_guard<std::mutex> guard(_zip_archives_mutex);
            auto it = _zip_archives.find(zip_filename);
            if (it == _zip_archives.end()) {
                if (_zip_archives.size() > 100) {
                    auto it = _zip_archives.find(_fifo.front());
                    zip_close(it->second);
                    _zip_archives.erase(it);
                    _fifo.pop();
                }
                zip_archive = zip_open(zip_filename.c_str(), ZIP_RDONLY, NULL);
                if (!zip_archive) error("can't read: " << zip_filename << " : " << get_error(errno));
                _zip_archives.insert(make_pair(zip_filename, zip_archive));
                _fifo.push(zip_filename);
            } else {
                zip_archive = it->second;
            }
        }
        return zip_archive;
    }
private:
    ZipArchive() {}
    unordered_map<string, zip_t*> _zip_archives;
    queue<string> _fifo;
    mutex _zip_archives_mutex;
};

struct Reader {
    virtual ssize_t getline(char** line) = 0;
    virtual string filename() = 0;
    virtual ssize_t fread(char* data, size_t length) = 0;
    virtual ~Reader() {}
};

static const size_t BUF_SIZE = 64 * 1024;

ssize_t get_index(char* buf, size_t n, char c) {
    for (size_t i = 0; i < n; ++i) {
        if (buf[i] == c) return i;
    }
    return -1;
}

ssize_t read_line(char** buf, size_t* buf_size, size_t* begin_idx, char** line, Reader* reader) {
    //if buffer is empty read up to buf size into it
    //if cannot read then return -1
    //if buf has data then try to read a line from last position
    //if no line read then we have a partial line
    //copy into
    //read up to buf size
    ssize_t bytes_read = 0;
    size_t line_size = 0;
    ssize_t end_idx = 0;
    if (!*buf) {
        *buf = static_cast<char*>(::malloc(BUF_SIZE));
        //first time only
        bytes_read = reader->fread(*buf, BUF_SIZE);
        if (bytes_read <= 0) return bytes_read;
        *begin_idx = 0;
        *buf_size = bytes_read;
    }
    //buf has data already, try to read a line
    end_idx = get_index(*buf + *begin_idx, *buf_size - *begin_idx, '\n');
    int begin_inc = 1;
    if (end_idx > 0 && (*buf + *begin_idx)[end_idx - 1] == '\r') { // windows cr/lf
        end_idx -= 1;  // end_idx now points to \r in \r\n
        begin_inc = 2;  // next line begins after end_idx + 2 (for \r\n)
    }

    if (end_idx >= 0) {
        //found a line. update begin index and set line ptr to beginning of this line
        line_size = end_idx;
        (*buf)[*begin_idx + end_idx] = '\0';  // replace \r or \n with \0 to end line
        *line = *buf + *begin_idx;
        *begin_idx += (end_idx + begin_inc);
        return line_size;
    } else {
        //we read a partial line
        //copy partial line to new buffer
        char* tmp = static_cast<char*>(::malloc(BUF_SIZE));
        size_t partial_str_size = BUF_SIZE - *begin_idx;
        strncpy(tmp, *buf + *begin_idx, partial_str_size);
        ::free(*buf);
        *buf = tmp;
        *begin_idx = 0;
        bytes_read = reader->fread(*buf + partial_str_size, BUF_SIZE - partial_str_size);
        if (bytes_read <= 0) return bytes_read;
        *buf_size = bytes_read + partial_str_size;
        return read_line(buf, buf_size, begin_idx, line, reader);
    }
}
    

class ZipReader: public Reader {
public:
    ZipReader(const std::string& filename):
    _filename(filename),
    _zip_file(nullptr),
    _buf(nullptr),
    _buf_idx(0),
    _buf_size(0) {
        std::size_t i = filename.find(':');
        auto zip_filename = filename.substr(0, i);
        auto inner_filename = filename.substr(i + 1);
        ZipArchive& archive = ZipArchive::get_instance();
        zip_t* zip_archive = archive.get_archive(zip_filename);
        _zip_file = zip_fopen(zip_archive, inner_filename.c_str(), ZIP_FL_ENC_GUESS);
        if (!_zip_file) error("can't read: " << inner_filename << " from " << filename << " : " << get_error(errno));
    }
    
    string filename() override { return _filename; }
    
    ssize_t getline(char** line) override {
        return read_line(&_buf, &_buf_size, &_buf_idx, line, this);
    }
    
    ssize_t fread(char* buf, size_t buf_size) override {
        return zip_fread(_zip_file, buf, buf_size);
    }
    
    ~ZipReader() {
        if (_zip_file) zip_fclose(_zip_file);
        _zip_file = nullptr;
        if (_buf) ::free(_buf);
    }
    
private:
    string _filename;
    zip_file_t* _zip_file;
    char* _buf;
    size_t _buf_idx;
    size_t _buf_size;
};

class FileReader: public Reader {
public:
    FileReader(const std::string& filename):
        _filename(filename),
        _file(::fopen(filename.c_str(), "r")),
        _buf(nullptr),
        _buf_idx(0),
        _buf_size(0)
    {}
    
    string filename() override {
        return _filename;
    }

    ssize_t getline(char** line) override {
        return read_line(&_buf, &_buf_size, &_buf_idx, line, this);
    }
    
    ssize_t fread(char* buf, size_t buf_size) override {
        if (feof(_file)) return -1;
        if (ferror(_file)) error("error reading file");
        size_t elems_read = ::fread(buf, sizeof(char), ::floor(buf_size / sizeof(char)), _file);
        return elems_read * sizeof(char);
    }
    
    ~FileReader() {
        if (_file) ::fclose(_file);
        _file = nullptr;
        if (_buf) ::free(_buf);
    }
    
private:
    string _filename;
    FILE* _file;
    char* _buf;
    size_t _buf_idx;
    size_t _buf_size;
};



bool read_csv_file(Reader* reader,
                   const std::vector<int>& col_indices,
                   const std::vector<std::string>& dtypes,
                   char separator,
                   int skip_rows,
                   int max_rows,
                   vector<void*>& output) {
    
    int row_num = 0;
    output.resize(dtypes.size());
    for (size_t i = 0; i < dtypes.size(); ++i) {
        output[i] = create_vector(dtypes[i], max_rows);
    }
    
    bool more_to_read = true;
    for (;;) {
        char* line = nullptr;
        ssize_t line_size = reader->getline(&line);
        if (line_size <= 0) {
            //eof or error.  ::getline returns zero in both cases, zip_fread returns -1 for error, 0 for eof
            more_to_read = false;
            break;
        }
        // cout << "row num: " << row_num << " len: " << strlen(line) << " " << line << endl;
        row_num++;
 
        if (row_num <= skip_rows) continue;
        
        if ((max_rows != 0) && (row_num > max_rows)) break;
        auto fields = tokenize_line(line, separator, col_indices);
        if (!fields.size()) continue; // empty line
        if (fields.size() != dtypes.size()) {
            //replace nulls we added with separator so we can print out the line
            string _line(line, line_size);
            std::replace(_line.begin(), _line.end(), '\0', separator);
            error(reader->filename() << " found " << fields.size() << " " << " fields on row: " << row_num
                  << " line: " << _line << " but dtypes arg length was " << dtypes.size() << endl)
        }
        add_line(fields, dtypes, output);
    }
    return more_to_read;
}

void test_csv_reader() {
    ifstream istr("/Users/sal/tmp/test.csv", ios_base::in);
    auto dtypes = vector<string>{
        "M8[ms]",
        "S10",
        "i4",
        "f8",
        "i1"};
    vector<void*> output(dtypes.size());
    bool more_to_read = false;
    auto vec1 = reinterpret_cast<vector<string>*>(output[0]);
    auto vec2 = reinterpret_cast<vector<string>*>(output[1]);
    cout << "row1: " << (*vec1)[0] << " " << (*vec2)[0] << "\n"
         << "row2: " << (*vec1)[1] << " " << (*vec2)[1] << "\n"
         << "more_to_read: " << more_to_read << endl;
    istr.close();
}



bool read_csv(const std::string& filename,
              const std::vector<int>& col_indices,
              const std::vector<std::string>& dtypes,
              char separator,
              int skip_rows,
              int max_rows,
              std::vector<void*>& output) {
    bool more_to_read = false;
    std::size_t i = filename.find(':');
    Reader* reader = NULL;
    if (i == filename.npos) {
        reader = new FileReader(filename);
        bool tmp = read_csv_file(reader, col_indices, dtypes, separator, skip_rows, max_rows, output);
        if (tmp) more_to_read = true;
        delete reader;
    } else {
        reader = new ZipReader(filename);
        bool tmp = read_csv_file(reader, col_indices, dtypes, separator, skip_rows, max_rows, output);
        if (tmp) more_to_read = true;
        delete reader;
    }
    return more_to_read;
}


void test_csv_reader2() {
    cout << "starting" << endl;
    vector<void*> output(2);
    bool more_to_read = read_csv("/Users/sal/tmp/test.csv",
                                 {15, 18, 20},
                                 {"f4", "f4", "i4"},
                                 ',',
                                 1,
                                 0,
                                 output);
    auto vec1 = static_cast<vector<float>*>(output[0]);
    cout << "num_cols: " << output.size() << " num rows: " << vec1->size() << " more_to_read: " << more_to_read
         << " first entry: " << (*vec1)[0] << endl;
}

void test_csv_reader_zip() {
    for (int j=0; j < 100000; ++j) {
        cout << "starting" << endl;
        vector<void*> output(2);
        bool more_to_read = read_csv("/Users/sal/tmp/algo/20220316.zip:20220316/A/AAPL.csv",
                                     {2, 9, 18, 27, 35, 48, 49},
                                     {"S5", "f4", "f4", "f4", "f4", "i4", "i4"},
                                     ',',
                                     1,
                                     0,
                                     output);
        auto vec1 = static_cast<vector<float>*>(output[1]);
        cout << "num_cols: " << output.size() << " num rows: " << vec1->size() << " more_to_read: " << more_to_read
        << " first entry: " << (*vec1)[0] << endl;
        delete static_cast<vector<string>*>(output[0]);
        for (size_t i=1; i < output.size(); ++i) {
            if (i != 0 && i != 5) {
                delete static_cast<vector<float>*>(output[i]);
            }
        }
        delete static_cast<vector<int32_t>*>(output[5]);
    }
    cout << "done" << endl;
}

