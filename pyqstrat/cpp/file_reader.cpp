#ifndef _WIN32

#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/lzma.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#endif

#include <fstream>
#include "utils.hpp"
#include "zip_reader.hpp"


using namespace std;

#ifndef _WIN32

namespace io = boost::iostreams;

#endif

#include "file_reader.hpp"

using namespace std;

class StreamHolder : public LineReader {
public:
    inline StreamHolder(shared_ptr<istream> file,
                        shared_ptr<streambuf> buf = nullptr,
                        shared_ptr<istream> istream = nullptr) :
    _file(file),
    _buf(buf),
    _istream(istream) {}
    
    bool call(std::string& line) override {
        std::istream& istr = std::getline(*_istream, line);
        if (istr) return true;
        return false;
    }
    virtual ~StreamHolder() = default;
private:
    std::shared_ptr<std::istream> _file;
    std::shared_ptr<std::streambuf> _buf;
    std::shared_ptr<std::istream> _istream;
};

shared_ptr<LineReader> TextFileDecompressor::call(const string& input_filename, const string& compression) {
    if (!compression.empty()) {
#ifdef _WIN32
        error("Reading compressed marketdata files not currently supported on windows");
#endif
    } else {
        std::shared_ptr<ifstream> file = shared_ptr<ifstream>(new ifstream(input_filename, std::ios_base::in));
        return shared_ptr<StreamHolder>(new StreamHolder(nullptr, nullptr, file));
    }
#ifndef _WIN32
    std::shared_ptr<ifstream> file = shared_ptr<ifstream>(new ifstream(input_filename, std::ios_base::in | std::ios_base::binary));
    auto buf = shared_ptr<io::filtering_streambuf<io::input>>(new io::filtering_streambuf<io::input>());
    if (compression == "gzip") buf->push(boost::iostreams::gzip_decompressor());
    else if (compression == "bz2") buf->push(io::bzip2_decompressor());
    else if (compression == "xz") buf->push(io::lzma_decompressor());
    else error("invalid compression: " << compression);
    buf->push(*file);
    auto istr = shared_ptr<istream>(new istream(buf.get()));
    return shared_ptr<LineReader>(new StreamHolder(file, buf, istr));
#endif
}

class ZipLineReader : public LineReader {
public:
    ZipLineReader(const string& filename) :
    _zr(filename),
    _curr_file_index(0),
    _curr_file_reader(nullptr),
    _filenames(_zr.get_filenames()) {
        if (!_filenames.empty()) _curr_file_reader = _zr.get_file_reader(_filenames[0]);
    }
    
    bool call(std::string& line) override {
        if (!_curr_file_reader) return false;
        std::istream& istr = std::getline(*_curr_file_reader, line);
        if (istr) return true;
        _curr_file_index++;
        if (_curr_file_index == _filenames.size()) return false;
        _curr_file_reader = _zr.get_file_reader(_filenames[_curr_file_index]);
        string dummy;
        // Throw away header line
        std::getline(*_curr_file_reader, dummy);
        return call(line);
    }
 private:
    ZipReader _zr;
    size_t _curr_file_index;
    shared_ptr<istream> _curr_file_reader;
    vector<string> _filenames;
};


shared_ptr<LineReader> ZipFileReader::call(const string& input_filename, const string& compression) {
    return shared_ptr<LineReader>(new ZipLineReader(input_filename));
}

void test_file_reader() {
    ZipLineReader zlreader("/tmp/tmp2/20181228.zip");
    int i = 0;
    string line;
    for (;;) {
        if (!zlreader.call(line)) break;
        //cout << i << ":" << line << endl;
        if (i % 100000 == 0) cout << i << endl;
        i++;
    }
}

