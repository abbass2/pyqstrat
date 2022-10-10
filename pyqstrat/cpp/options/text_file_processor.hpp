#ifndef __text_file_processor_hpp
#define __text_file_processor_hpp

#include <string>
#include <stdexcept>
#include <regex>
#include <cmath>
#include <istream>
#include <fstream>


#include "pq_types.hpp"

struct PriceQtyMissingDataHandler :  public MissingDataHandler {
    bool call(std::shared_ptr<Record> record) override;
};

class PrintBadLineHandler : public BadLineHandler {
public:
    PrintBadLineHandler(bool raise = false);
    std::shared_ptr<Record> call(int line_number, const std::string& line, const std::exception& ex) override;
private:
    bool _raise;
};

class RegExLineFilter : public LineFilter {
public:
    RegExLineFilter(const std::string& pattern);
    bool call(const std::string& line) override;
private:
    std::regex _pattern;
};

class SubStringLineFilter : public LineFilter {
public:
    SubStringLineFilter(const std::vector<std::string>& patterns);
    bool call(const std::string& line) override;
private:
    std::vector<std::string> _patterns;
};

class IsFieldInList : public CheckFields {
public:
    IsFieldInList(int flag_idx, const std::vector<std::string>& flag_values);
    bool call(const std::vector<std::string>& fields) override;
private:
    int _flag_idx;
    std::vector<std::string> _flag_values;
};


class TextFileProcessor : public FileProcessor {
public:
    TextFileProcessor(
                      RecordGenerator* record_generator,
                      LineFilter* line_filter,
                      RecordParser* record_parser,
                      BadLineHandler* bad_line_handler,
                      RecordFilter* record_filter,
                      MissingDataHandler* missing_data_handler,
                      std::vector<Aggregator*> aggregators,
                      int skip_rows);
    int call(const std::string& input_filename, const std::string& compression);
private:
    Function<std::shared_ptr<LineReader>(const std::string&, const std::string&)>* _record_generator;
    Function<bool (const std::string&)>* _line_filter;
    RecordParser* _record_parser;
    Function<std::shared_ptr<Record> (int, const std::string&, const std::exception&)>* _bad_line_handler;
    Function<bool (const Record&)>* _record_filter;
    Function<bool (std::shared_ptr<Record>)>* _missing_data_handler;
    std::vector<Aggregator*> _aggregators;
    int _skip_rows;
};

#endif
