#ifndef text_file_parsers_hpp
#define text_file_parsers_hpp

#include <string>
#include <vector>

#include "utils.hpp"
#include "types.hpp"

#include <boost/iostreams/filtering_stream.hpp>


typedef std::function<bool(const std::vector<std::string>&)> IsRecordFunc;
typedef std::function<int64_t (const std::string&)> TimestampParser;

class FormatTimestampParser {
public:
    FormatTimestampParser(int64_t base_date, const std::string& time_format = "%H:%M:%S", bool micros = false);
    int64_t operator()(const std::string& time);
private:
    int64_t _base_date;
    std::string _time_format;
    bool _micros;
};

inline int64_t fast_time_milli_parser(const std::string& time) {
    if (time.size() != 12 || time[2] != ':' || time[5] != ':' || time[8] != '.') error("timestamp not in HH:MM:SS format: " << time)
    int hours = str_to_int(time.substr(0, 2).c_str());
    int minutes = str_to_int(time.substr(3, 2).c_str());
    int seconds = str_to_int(time.substr(6, 2).c_str());
    int millis = str_to_int(time.substr(9, 3).c_str());
    return static_cast<int64_t>((hours * 60 * 60 + minutes * 60 + seconds) * 1000 + millis);
}

inline int64_t fast_time_micro_parser(const std::string& time) {
    if (time.size() != 15 || time[2] != ':' || time[5] != ':' || time[8] != '.') error("timestamp not in HH:MM:SS format: " << time)
    int hours = str_to_int(time.substr(0, 2).c_str());
    int minutes = str_to_int(time.substr(3, 2).c_str());
    int seconds = str_to_int(time.substr(5, 2).c_str());
    int micros = str_to_int(time.substr(9, 6).c_str());
    return static_cast<int64_t>((hours * 60 * 60 + minutes * 60 + seconds) * 1000 + micros);
}

class TextQuoteParser {
public:
    TextQuoteParser(IsRecordFunc is_quote,
                    int64_t base_date,
                    int timestamp_idx,
                    int bid_offer_idx,
                    int price_idx,
                    int qty_idx,
                    const std::vector<int>& id_field_indices,
                    const std::vector<int>& meta_field_indices,
                    TimestampParser timestamp_parser,
                    const std::string& bid_str,
                    const std::string& offer_str,
                    float price_multiplier = 1.0,
                    bool strip_id = true,
                    bool strip_meta = true);
    
    std::shared_ptr<QuoteRecord> operator()(const std::vector<std::string>& fields);
private:
    IsRecordFunc _is_quote;
    int64_t _base_date;
    int _timestamp_idx;
    int _bid_offer_idx;
    int _price_idx;
    int _qty_idx;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    TimestampParser _timestamp_parser;
    std::string _bid_str;
    std::string _offer_str;
    float _price_multiplier;
    bool _strip_id;
    bool _strip_meta;
};

class TextTradeParser {
public:
    TextTradeParser(IsRecordFunc is_trade,
                    int64_t base_date,
                    int timestamp_idx,
                    int price_idx,
                    int qty_idx,
                    const std::vector<int>& id_field_indices,
                    const std::vector<int>& meta_field_indices,
                    TimestampParser timestamp_parser,
                    float price_multiplier = 1,
                    bool strip_id = true,
                    bool strip_meta = true);
    
    std::shared_ptr<TradeRecord> operator()(const std::vector<std::string>& fields);
    
private:
    IsRecordFunc _is_trade;
    int64_t _base_date;
    int _timestamp_idx;
    int _price_idx;
    int _qty_idx;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    TimestampParser _timestamp_parser;
    float _price_multiplier;
    bool _strip_id;
    bool _strip_meta;
};

class TextOpenInterestParser {
public:
    TextOpenInterestParser(IsRecordFunc is_open_interest,
                           int64_t base_date,
                           int timestamp_idx,
                           int qty_idx,
                           const std::vector<int>& id_field_indices,
                           const std::vector<int>& meta_field_indices,
                           TimestampParser timestamp_parser,
                           bool strip_id = true,
                           bool strip_meta = true);
    
    std::shared_ptr<OpenInterestRecord> operator()(const std::vector<std::string>& fields);
    
private:
    IsRecordFunc _is_open_interest;
    int64_t _base_date;
    int _timestamp_idx;
    int _qty_idx;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    TimestampParser _timestamp_parser;
    bool _strip_id;
    bool _strip_meta;
};

class TextOtherParser {
public:
    TextOtherParser(IsRecordFunc is_other,
                    int64_t base_date,
                    int timestamp_idx,
                    const std::vector<int>& id_field_indices,
                    const std::vector<int>& meta_field_indices,
                    TimestampParser timestamp_parser,
                    bool strip_id = true,
                    bool strip_meta = true);
    
    std::shared_ptr<OtherRecord> operator()(const std::vector<std::string>& fields);

private:
    IsRecordFunc _is_other;
    int64_t _base_date;
    int _timestamp_idx;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    TimestampParser _timestamp_parser;
    bool _strip_id;
    bool _strip_meta;
};

class TextRecordParser {
public:
    
    TextRecordParser(std::shared_ptr<TextQuoteParser> quote_parser, std::shared_ptr<TextTradeParser> trade_parser,
                     std::shared_ptr<TextOpenInterestParser> open_interest_parser, std::shared_ptr<TextOtherParser> other_parser,
                     char separator = ',' );
    
    std::shared_ptr<Record> operator()(const std::string& line);
    
private:
    std::shared_ptr<TextQuoteParser> _quote_parser;
    std::shared_ptr<TextTradeParser> _trade_parser;
    std::shared_ptr<TextOpenInterestParser> _open_interest_parser;
    std::shared_ptr<TextOtherParser> _other_parser;
    std::vector<std::string> _headers;
    char _separator;
};

#endif /* text_file_parsers_hpp */

