#include <algorithm>


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

/*vector<string> intersection(const vector<string> &v1_, const vector<string> &v2_){
    vector<string> v3;
    vector<string> v1 = v1_;
    vector<string> v2 = v2_;
    sort(v1.begin(), v1.end());
    sort(v2.begin(), v2.end());
    set_intersection(v1.begin(),v1.end(), v2.begin(),v2.end(), back_inserter(v3));
    return v3;
} */



class ZipLineReader : public LineReader {
public:
    ZipLineReader(const string& archive_filename, const vector<string>& patterns) :
    _zr(archive_filename),
    _curr_file_index(0),
    _curr_file_reader(nullptr) {
        vector<string> archive_filenames = _zr.get_filenames();
        
        for (const string& archive_filename : archive_filenames) {
            for (const string& pattern : patterns) {
                if (archive_filename.find(pattern) != string::npos) {
                    _filenames.push_back(archive_filename);
                    break;
                }
            }
        }
        if (_filenames.empty()) return;
        _curr_file_reader = _zr.get_file_reader(_filenames[0]);
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

ZipFileReader::ZipFileReader(const map<string, vector<string>>& archive_to_filenames) :
_archive_to_filenames(archive_to_filenames) {}

shared_ptr<LineReader> ZipFileReader::call(const string& input_filename, const string& compression) {
    auto p = _archive_to_filenames.find(input_filename);
    if (p == _archive_to_filenames.end()) error(input_filename << " not found in zip");
    return shared_ptr<LineReader>(new ZipLineReader(input_filename, p->second));
    
}

void test_zip_file_reader() {
    string archive_name = "/Users/sal/tmp/algoseek/emini_options_1min_bars/2015/20150102.zip";
    auto filenames = vector<string>{"EW2F5/EW2F5.P1600.csv", "EW2F5/EW2F5.C2230.csv"};
    map<string, vector<string>> archive_to_filenames;
    archive_to_filenames.insert(make_pair(archive_name, filenames));
    ZipFileReader zip_file_reader(archive_to_filenames);
    std::shared_ptr<LineReader> line_reader = zip_file_reader.call(archive_name, "zip");
    int i = 0;
    string line;
    for (;;) {
        if (!line_reader->call(line)) break;
        cout << i << ":" << line << endl;
        if (i % 100000 == 0) cout << i << endl;
        i++;
    }
}

