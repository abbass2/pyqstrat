//
//  utils.hpp
//  py_c_test
//
//  Created by Sal Abbasi on 9/12/22.
//

#ifndef utils_hpp
#define utils_hpp

#include <string.h>
#include <sstream>

#define error(msg) \
{ \
std::ostringstream os; \
os << msg << " file: " << __FILE__ << " line: " << __LINE__ ; \
throw std::runtime_error(os.str()); \
}

#endif /* utils_hpp */
