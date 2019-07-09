#ifndef __file_reader_hpp
#define __file_reader_hpp

#include <string>
#include "pq_types.hpp"

struct TextFileDecompressor : public RecordGenerator {
    std::shared_ptr<LineReader> call(const std::string& filename, const std::string& compression) override;
};

struct ZipFileReader : public RecordGenerator {
    std::shared_ptr<LineReader> call(const std::string& filename, const std::string& compression) override;
};

void test_file_reader();

#endif
