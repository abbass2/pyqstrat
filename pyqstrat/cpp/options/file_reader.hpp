#ifndef __file_reader_hpp
#define __file_reader_hpp

#include <string>
#include <map>
#include "pq_types.hpp"


struct TextFileDecompressor : public RecordGenerator {
    std::shared_ptr<LineReader> call(const std::string& filename, const std::string& compression) override;
};

class ZipFileReader : public RecordGenerator {
public:
    explicit ZipFileReader(const std::map<std::string, std::vector<std::string>>& archive_to_filenames);
    std::shared_ptr<LineReader> call(const std::string& filename, const std::string& compression) override;
private:
    std::map<std::string, std::vector<std::string>> _archive_to_filenames;
};

void test_zip_file_reader();

#endif
