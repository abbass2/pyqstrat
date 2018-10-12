#ifndef utils_hpp
#define utils_hpp

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>

#define error(msg) \
{ \
std::ostringstream os; \
os << msg << " file: " << __FILE__ << " line: " << __LINE__ ; \
throw std::runtime_error(os.str()); \
}

template<class T, class U> bool assert_present_in_vector(const T& value, const U& vec, bool raise = true) {
    bool found = (find(vec.begin(), vec.end(), value) != vec.end());
    if (found) return true;
    if (!raise) return false;
    error(value << " not found");
}

template<typename InputIt> std::string join_string(InputIt first, InputIt last, const std::string& separator = ", ", const std::string& concluder = "") {
    if (first == last) return concluder;
    std::stringstream ss;
    ss << *first;
    ++first;
    while (first != last) {
        ss << separator;
        ss << *first;
        ++first;
    }
    ss << concluder;
    return ss.str();
}

std::string join_fields(const std::vector<std::string>& fields, const std::vector<int>& indices, char separator, bool strip);

int64_t str_to_timestamp(const std::string& str, const std::string& date_format, bool micros = false);

std::vector<std::string> tokenize(const char* str, char separator);

float str_to_float(const char* str, char decimal_point = '.', char thousands_separator = ',');
int str_to_int(const char* str, char thousands_separator = ',');

inline float str_to_float(const std::string& str, char decimal_point = '.', char thousands_separator = ',') {
    return str_to_float(str.c_str(), decimal_point, thousands_separator);
}

#endif /* utils_hpp */

