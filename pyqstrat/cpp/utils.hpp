#ifndef utils_hpp
#define utils_hpp

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>


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

// The following string trim functions are from https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
// trim from start (in place)
inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return ((ch != ' ') & (ch != '\n') & (ch != '\r'));
    }));
}

// trim from end (in place)
inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return ((ch != ' ') & (ch != '\n') & (ch != '\r'));
    }).base(), s.end());
}

// trim from both ends (in place)
inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

// trim from start (copying)
inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

#endif /* utils_hpp */

