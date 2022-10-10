#ifndef zip_reader_hpp
#define zip_reader_hpp

#include <string>
#include <vector>
#include <memory>
#include <zip.h>

class ZipReader final {
public:
    ZipReader(const std::string& filename);
    std::vector<std::string> get_filenames();
    std::shared_ptr<std::istream> get_file_reader(const std::string& filename);
    ~ZipReader();
    
private:
    zip_t* _ziparchive;
};

void test_zip_reader();

#endif


