#include <vector>
#include <cmath>
#include <string>
#include <boost/algorithm/string.hpp>

#include "date.hpp"
#include "utils.hpp"


using namespace std;

float str_to_float(const char* str, char decimal_point, char thousands_separator) {
    // convert a string to a float
    float result = 0;
    float sign = *str == '-' ? static_cast<void>(str++), -1 : 1;
    while ((*str >= '0' && *str <= '9') || (*str == thousands_separator)) {
        if (*str == thousands_separator) {
            str++;
            continue;
        }
        result *= 10;
        result += *str - '0';
        str++;
    }
    
    float multiplier = 0.1;
    if (*str == decimal_point) {
        str++;
        while (*str >= '0' && *str <= '9') {
            result += (*str - '0') * multiplier;
            multiplier /= 10;
            str++;
        }
    }
    
    float power = 0;
    result *= sign;
    if (*str == 'e' || *str == 'E') {
        str++;
        float powerer = *str == '-'? static_cast<void>(str++), 0.1 : 10;
        
        while ((*str >= '0') && (*str <= '9')) {
            power *= 10;
            power += *str - '0';
            str++;
        }
        result *= pow(powerer, power);
    }
    return result;
}

int str_to_int(const char* str, char thousands_separator) {
    // convert a string to a float
    int result = 0;
    int sign = *str == '-' ? static_cast<void>(str++), -1 : 1;
    while ((*str >= '0' && *str <= '9') || (*str == thousands_separator)) {
        if (*str == thousands_separator) {
            str++;
            continue;
        }
        result *= 10;
        result += *str - '0';
        str++;
    }
    result *= sign;
    return result;
}

vector<string> tokenize(const char* str, const char separator) {
    vector<string> tokens;
    const char* token_start = str;
    while (true) {
        if ((*str == separator) || (*str == '\0')) {
            tokens.push_back(string(token_start, str));
            token_start = str + 1;
            if (*str == '\0') break;
        }
        str++;
    }
    return tokens;
}

std::string join_fields(const vector<string>& fields, const vector<int>& indices, char separator, bool strip) {
    stringstream ss;
    bool first = true;
    for (auto idx : indices) {
        const std::string& str = fields[idx];
        if (!str.size()) continue;
        if (first) {
            if (strip) {
                std::string stripped = boost::trim_copy_if(str, boost::is_any_of("\n\r "));
                if (stripped.size()) ss << stripped;
            } else {
                ss << str;
            }
            first = false;
        } else {
            if (strip) {
                std::string stripped = boost::trim_copy_if(str, boost::is_any_of("\n\r "));
                if (stripped.size()) ss << separator << stripped;
            } else {
                ss << separator << str;
            }
        }
    }
    return ss.str();
}

int64_t str_to_timestamp(const std::string& str, const std::string& date_format, bool micros) {
    date::sys_time<std::chrono::milliseconds> tp_millis;
    date::sys_time<std::chrono::microseconds> tp_micros;
    std::istringstream ss{str};
    if (micros) {
        ss >> date::parse(date_format, tp_micros);
        if (ss.fail()) error("Could not parse: " << str << " with format: " << date_format);
        return tp_micros.time_since_epoch().count();
    } else {
        ss >> date::parse(date_format, tp_millis);
        if (ss.fail()) error("Could not parse: " << str << " with format: " << date_format);
        return tp_millis.time_since_epoch().count();
    }
}

