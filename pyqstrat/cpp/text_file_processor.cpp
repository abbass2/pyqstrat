#include <stdexcept>
#include <iostream>
#include <regex>
#include <fstream>      // std::ifstream
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/lzma.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include "utils.hpp"
#include "text_file_processor.hpp"

using namespace std;
namespace io = boost::iostreams;

void price_qty_missing_data_handler(shared_ptr<Record> record) {
    shared_ptr<QuoteRecord> quote = dynamic_pointer_cast<QuoteRecord>(record);
    if (quote) {
        if (quote->qty == 0) quote->qty = NAN;
        if (quote->price == 0) quote->price = NAN;
        return;
    }
    shared_ptr<TradeRecord> trade = dynamic_pointer_cast<TradeRecord>(record);
    if (trade) {
        if (trade->qty == 0) trade->qty = NAN;
        if (trade->price == 0) trade->price = NAN;
        return;
    }
    shared_ptr<OpenInterestRecord> oi = dynamic_pointer_cast<OpenInterestRecord>(record);
    if (oi) {
        if (oi->qty == 0) oi->qty = NAN;
        return;
    }
}

PrintBadLineHandler::PrintBadLineHandler(bool raise) : _raise(raise) {}

shared_ptr<Record> PrintBadLineHandler::operator()(int line_number, const std::string& line, const std::exception& ex) {
    cerr << "parse error: " << ex.what() << " line number: " << line_number << " line: " << line << endl;
    if (_raise) throw ex;
    return nullptr;
}

shared_ptr<StreamHolder> text_file_decompressor(const string& input_filename, const string& compression) {
    std::shared_ptr<ifstream> file = shared_ptr<ifstream>(new ifstream(input_filename, std::ios_base::in | std::ios_base::binary));
    auto buf = shared_ptr<io::filtering_streambuf<io::input>>(new io::filtering_streambuf<io::input>());
    if (compression == "gzip") buf->push(boost::iostreams::gzip_decompressor());
    else if (compression == "bz2") buf->push(io::bzip2_decompressor());
    else if (compression == "xz") buf->push(io::lzma_decompressor());
    else error("invalid compression: " << compression);
    buf->push(*file);
    auto istr = shared_ptr<istream>(new istream(buf.get()));
    return shared_ptr<StreamHolder>(new StreamHolder(buf, file, istr));
}

StreamHolder::~StreamHolder() {
    //if (_file) _file->close();
}

RegExLineFilter::RegExLineFilter(const std::string& pattern) : _pattern (pattern) {}
    
bool RegExLineFilter::operator()(const std::string& line) {
    return std::regex_match(line, _pattern);
}

SubStringLineFilter::SubStringLineFilter(const vector<std::string>& patterns) :
    _patterns(patterns) {}

bool SubStringLineFilter::operator()(const std::string& line) {
    int size = static_cast<int>(line.size());
    for (int i = 0; i < size; ++i) {
        for (auto pattern : _patterns) {
            bool found = true;
            int k = 0;
            for (int j = 0; j < static_cast<int>(pattern.size()); ++j) {
                if (line[i + k] != pattern[j] || (i + k) == size) {
                    found = false;
                    break;
                }
                ++k;
            }
            if (found) return true;
        }
    }
    return false;
}

bool is_field_in_list(const vector<string>& fields, int flag_idx, const std::vector<string>& flag_values) {
    const std::string& val = fields[flag_idx];
    return (std::find(flag_values.begin(), flag_values.end(), val) != flag_values.end());
}

TextFileProcessor::TextFileProcessor(
                                    std::function<shared_ptr<StreamHolder>(const std::string&, const std::string&)> record_generator,
                                    std::function<bool (const std::string&)> line_filter,
                                    std::function<shared_ptr<Record> (const std::string&)> record_parser,
                                    std::function<shared_ptr<Record> (int, const std::string&, const std::exception&)> bad_line_handler,
                                    std::function<bool (const Record&)> record_filter,
                                    std::function<void (shared_ptr<Record>)> missing_data_handler,
                                    std::function<void (const QuoteRecord&, int)> quote_aggregator,
                                    std::function<void (const TradeRecord&, int)> trade_aggregator,
                                    std::function<void (const OpenInterestRecord&, int)> open_interest_aggregator,
                                    std::function<void (const OtherRecord&, int)> other_aggregator,
                                    int skip_rows) :
_record_generator(record_generator),
_line_filter(line_filter),
_record_parser(record_parser),
_bad_line_handler(bad_line_handler),
_record_filter(record_filter),
_missing_data_handler(missing_data_handler),
_quote_aggregator(quote_aggregator),
_trade_aggregator(trade_aggregator),
_open_interest_aggregator(open_interest_aggregator),
_other_aggregator(other_aggregator),
_skip_rows(skip_rows){}

int TextFileProcessor::operator()(const std::string& input_filename, const std::string& compression) {
    cout << "processing:" << input_filename << " process id: " << ::getpid() << endl;
    shared_ptr<StreamHolder> istr = _record_generator(input_filename, compression);
    string line;
    int line_number = 0;
    while ((*istr)(line)) {
        line_number++;
        //if (line_number > 200000) break;
        if (line_number <= _skip_rows) continue;
        if (_line_filter && !_line_filter(line)) continue;
        auto record = shared_ptr<Record>();
        try {
           record  = _record_parser(line);
        } catch(const ParseException& ex) {
            record = _bad_line_handler(line_number, line, ex);
            if (!record) continue;
        }
        if (record == nullptr) continue;
        //if (line_number % 10000 == 0) cout << "got record from line: " << line_number << ":" << line << endl;
        if (_record_filter && !_record_filter(*record)) continue;
        
        if (_missing_data_handler) _missing_data_handler(record);
        auto quote = dynamic_pointer_cast<QuoteRecord>(record);
        if (quote && _quote_aggregator) _quote_aggregator(*quote, line_number);
        auto trade = dynamic_pointer_cast<TradeRecord>(record);
        if (trade && _trade_aggregator) _trade_aggregator(*trade, line_number);
        auto oi = dynamic_pointer_cast<OpenInterestRecord>(record);
        if (oi && _open_interest_aggregator) _open_interest_aggregator(*oi, line_number);
        auto other = dynamic_pointer_cast<OtherRecord>(record);
        if (other && _other_aggregator) _other_aggregator(*other, line_number);
    }
    return line_number;
}
