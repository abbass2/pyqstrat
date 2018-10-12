#ifndef __text_file_processor_hpp
#define __text_file_processor_hpp

#include <string>
#include <stdexcept>
#include <regex>
#include <cmath>
#include <istream>
#include <fstream>
#include <boost/iostreams/filtering_stream.hpp>

#include "types.hpp"

void price_qty_missing_data_handler(std::shared_ptr<Record> record);

class PrintBadLineHandler {
public:
    PrintBadLineHandler(bool raise = false);
    std::shared_ptr<Record> operator()(int line_number, const std::string& line, const std::exception& ex);
private:
    bool _raise;
};

class RegExLineFilter {
public:
    RegExLineFilter(const std::string& pattern);
    bool operator()(const std::string& line);
private:
    std::regex _pattern;
};

class SubStringLineFilter {
public:
    SubStringLineFilter(const std::vector<std::string>& patterns);
    bool operator()(const std::string& line);
private:
    std::vector<std::string> _patterns;
};

bool is_field_in_list(const std::vector<std::string>& fields, int flag_idx, const std::vector<std::string>& flag_values);


class StreamHolder {
public:
    inline StreamHolder(std::shared_ptr<boost::iostreams::filtering_streambuf<boost::iostreams::input>> buf, std::shared_ptr<std::istream> file, std::shared_ptr<std::istream> istream) :
    _buf(buf),
    _file(file),
    _istream(istream) {}
    bool operator()(std::string& line) {
        std::istream& istr = std::getline(*_istream, line);
        if (istr) return true;
        return false;
    }
    virtual ~StreamHolder();
private:
    std::shared_ptr<boost::iostreams::filtering_streambuf<boost::iostreams::input>> _buf;
    std::shared_ptr<std::istream> _file;
    std::shared_ptr<std::istream> _istream;
};

std::shared_ptr<StreamHolder> text_file_decompressor(const std::string& filename, const std::string& compression);

class TextFileProcessor {
public:
    TextFileProcessor(std::function<std::shared_ptr<StreamHolder>(const std::string&, const std::string&)> record_generator,
                      std::function<bool (const std::string&)> line_filter,
                      std::function<std::shared_ptr<Record>(const std::string&)> record_parser,
                      std::function<std::shared_ptr<Record>(int, const std::string&, const std::exception&)> bad_line_handler,
                      std::function<bool (const Record&)> record_filter,
                      std::function<void (std::shared_ptr<Record>)> missing_data_handler,
                      std::function<void (const QuoteRecord&, int)> quote_aggregator,
                      std::function<void (const TradeRecord&, int)> trade_aggregator,
                      std::function<void (const OpenInterestRecord&, int)> open_interest_aggregator,
                      std::function<void (const OtherRecord&, int)> other_aggregator,
                      int skip_rows = 1);
    int operator()(const std::string& input_filename, const std::string& compression);
private:
    std::function<std::shared_ptr<StreamHolder>(const std::string&, const std::string&)> _record_generator;
    std::function<bool (const std::string&)> _line_filter;
    std::function<std::shared_ptr<Record>(const std::string&)> _record_parser;
    std::function<std::shared_ptr<Record>(int, const std::string&, const std::exception&)> _bad_line_handler;
    std::function<bool (const Record&)> _record_filter;
    std::function<void (std::shared_ptr<Record>)> _missing_data_handler;
    std::function<void (const QuoteRecord&, int)> _quote_aggregator;
    std::function<void (const TradeRecord&, int)> _trade_aggregator;
    std::function<void (const OpenInterestRecord&, int)> _open_interest_aggregator;
    std::function<void (const OtherRecord&, int)> _other_aggregator;
    int _skip_rows;
};

#endif
