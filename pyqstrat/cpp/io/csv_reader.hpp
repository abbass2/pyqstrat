//
//  csv_reader.hpp
//  py_c_test
//
//  Created by Sal Abbasi on 9/12/22.
//

#ifndef csv_reader_hpp
#define csv_reader_hpp

#include <vector>
#include <string>

bool read_csv(const std::string& filename,
              const std::vector<int>& col_indices,
              const std::vector<std::string>& dtypes,
              char separator,
              int skip_rows,
              int max_rows,
              std::vector<void*>& output);

void test_csv_reader();
void test_csv_reader2();
void test_csv_reader_zip();
#endif /* csv_reader_hpp */
