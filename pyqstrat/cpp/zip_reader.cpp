#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <zip.h>
#include <iostream>
#include "zip_reader.hpp"
#include "utils.hpp"

using namespace std;

struct ZipBuf: std::streambuf {
    ZipBuf(zip_file_t* file): file_(file) {}
    ~ZipBuf() { zip_fclose(file_); }
private:
    zip_file_t* file_;
    enum { s_size = 8196 };
    char buffer_[s_size];
    int underflow() {
        zip_int64_t rc(zip_fread(this->file_, this->buffer_, s_size));
        this->setg(this->buffer_, this->buffer_, this->buffer_ + std::max(static_cast<zip_int64_t>(0), rc));
        return this->gptr() == this->egptr() ? traits_type::eof() : traits_type::to_int_type(*this->gptr());
    }
};

ZipReader::ZipReader(const std::string& filename) :
    _ziparchive(nullptr) {
    _ziparchive = zip_open(filename.c_str(), ZIP_RDONLY, NULL);
    if (!_ziparchive) error("Can't read: " << filename << " : " << strerror(errno));
}

ZipReader::~ZipReader() {
    if (_ziparchive) zip_close(_ziparchive);
    _ziparchive = nullptr;
}

vector<string> ZipReader::get_filenames() {
    vector<string> filenames;
    zip_int64_t num_entries = zip_get_num_entries(_ziparchive, 0);
    for (int i = 0; i < num_entries; ++i) {
        const char* name = zip_get_name(_ziparchive, i, ZIP_FL_ENC_GUESS);
        if((name[0] != '\0') && (name[strlen(name)-1] == '/')) continue; // Directory
        filenames.push_back(string(name));
    }
    return filenames;
}

void delete_istream(istream* p) {
    delete p->rdbuf();
    delete p;
    p = nullptr;
}

shared_ptr<istream> ZipReader::get_file_reader(const string& filename) {
    zip_file_t* zipfile = zip_fopen(_ziparchive, filename.c_str(), ZIP_FL_ENC_GUESS);
    auto zb = new ZipBuf(zipfile);
    auto in = shared_ptr<istream>(new istream(zb), delete_istream);
    return in;
}

void test_zip_reader() {
    ZipReader reader("/tmp/tmp2/20181228.zip");
    auto filenames = reader.get_filenames();
    for (auto filename : filenames) {
        cout << filename << endl;
    }
    
    for (int i = 0; i < 2; ++i) {
        shared_ptr<istream> is = reader.get_file_reader(filenames[i]);
        for (;;) {
            string line;
            std::istream& istr = std::getline(*is, line);
            if (!istr) break;
            cout << "got line: " << line << endl;
        }
        
    }
}
