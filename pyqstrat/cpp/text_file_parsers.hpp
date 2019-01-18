#ifndef text_file_parsers_hpp
#define text_file_parsers_hpp

#include <string>
#include <vector>
#include <map>

#include "utils.hpp"
#include "pq_types.hpp"

// #include <boost/iostreams/filtering_stream.hpp>



class FormatTimestampParser : public TimestampParser {
public:
    FormatTimestampParser(int64_t base_date, const std::string& time_format = "%H:%M:%S", bool micros = false);
    int64_t call(const std::string& time) override;
private:
    int64_t _base_date;
    std::string _time_format;
    bool _micros;
};

class FixedWidthTimeParser : public TimestampParser {
public:
    FixedWidthTimeParser(bool micros = false,
                         int years_start = -1,
                         int years_size = -1,
                         int months_start = -1,
                         int months_end = -1,
                         int days_start = -1,
                         int days_end = -1,
                         int hours_start = -1,
                         int hours_size = -1,
                         int minutes_start = -1,
                         int minutes_size = -1,
                         int seconds_start = -1,
                         int seconds_size = -1,
                         int millis_start = -1,
                         int millis_size = -1,
                         int micros_start = -1,
                         int micros_size = -1);
    int64_t call(const std::string& time) override;
private:
    int64_t parse_date(const std::string& date);
    int64_t parse_time(const std::string& time);
    bool _micros;
    int _years_start;
    int _years_size;
    int _months_start;
    int _months_size;
    int _days_start;
    int _days_size;
    int _hours_start;
    int _hours_size;
    int _minutes_start;
    int _minutes_size;
    int _seconds_start;
    int _seconds_size;
    int _millis_start;
    int _millis_size;
    int _micros_start;
    int _micros_size;
    std::map<std::string, int64_t> _parsed_date_cache;
};

class TextQuoteParser : public RecordFieldParser {
public:
    TextQuoteParser(CheckFields* is_quote,
                    int64_t base_date,
                    const std::vector<int>& timestamp_indices,
                    int bid_offer_idx,
                    int price_idx,
                    int qty_idx,
                    const std::vector<int>& id_field_indices,
                    const std::vector<int>& meta_field_indices,
                    const std::vector<TimestampParser*>& timestamp_parsers,
                    const std::string& bid_str,
                    const std::string& offer_str,
                    float price_multiplier = 1.0,
                    bool strip_id = true,
                    bool strip_meta = true);
    std::shared_ptr<Record> call(const std::vector<std::string>& fields) override;
private:
    CheckFields* _is_quote;
    int64_t _base_date;
    std::vector<int> _timestamp_indices;
    int _bid_offer_idx;
    int _price_idx;
    int _qty_idx;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    std::vector<TimestampParser*> _timestamp_parsers;
    std::string _bid_str;
    std::string _offer_str;
    float _price_multiplier;
    bool _strip_id;
    bool _strip_meta;
};

class TextQuotePairParser : public RecordFieldParser {
public:
    TextQuotePairParser(
                    CheckFields* is_quote_pair,
                    int64_t base_date,
                    const std::vector<int>& timestamp_indices,
                    int bid_price_idx,
                    int bid_qty_idx,
                    int ask_price_idx,
                    int ask_qty_idx,
                    const std::vector<int>& id_field_indices,
                    const std::vector<int>& meta_field_indices,
                    const std::vector<TimestampParser*>& timestamp_parsers,
                    float price_multiplier = 1.0,
                    bool strip_id = true,
                    bool strip_meta = true);
    
    std::shared_ptr<Record> call(const std::vector<std::string>& fields) override;
private:
    CheckFields* _is_quote_pair;
    int64_t _base_date;
    std::vector<int> _timestamp_indices;
    int _bid_price_idx;
    int _bid_qty_idx;
    int _ask_price_idx;
    int _ask_qty_idx;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    std::vector<TimestampParser*> _timestamp_parsers;
    float _price_multiplier;
    bool _strip_id;
    bool _strip_meta;
};


class TextTradeParser : public RecordFieldParser {
public:
    TextTradeParser(CheckFields* is_trade,
                    int64_t base_date,
                    const std::vector<int>& timestamp_indices,
                    int price_idx,
                    int qty_idx,
                    const std::vector<int>& id_field_indices,
                    const std::vector<int>& meta_field_indices,
                    const std::vector<TimestampParser*>& timestamp_parsers,
                    float price_multiplier = 1,
                    bool strip_id = true,
                    bool strip_meta = true);
    
    std::shared_ptr<Record> call(const std::vector<std::string>& fields) override;
    
private:
    CheckFields* _is_trade;
    int64_t _base_date;
    std::vector<int> _timestamp_indices;
    int _price_idx;
    int _qty_idx;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    std::vector<TimestampParser*> _timestamp_parsers;
    float _price_multiplier;
    bool _strip_id;
    bool _strip_meta;
};

class TextOpenInterestParser : public RecordFieldParser {
public:
    TextOpenInterestParser(CheckFields* is_open_interest,
                           int64_t base_date,
                           const std::vector<int>& timestamp_indices,
                           int qty_idx,
                           const std::vector<int>& id_field_indices,
                           const std::vector<int>& meta_field_indices,
                           const std::vector<TimestampParser*>& timestamp_parsers,
                           bool strip_id = true,
                           bool strip_meta = true);
    
    std::shared_ptr<Record> call(const std::vector<std::string>& fields) override;
    
private:
    CheckFields* _is_open_interest;
    int64_t _base_date;
    std::vector<int> _timestamp_indices;
    int _qty_idx;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    std::vector<TimestampParser*> _timestamp_parsers;
    bool _strip_id;
    bool _strip_meta;
};

class TextOtherParser : public RecordFieldParser {
public:
    TextOtherParser(CheckFields* is_other,
                    int64_t base_date,
                    const std::vector<int>& timestamp_indices,
                    const std::vector<int>& id_field_indices,
                    const std::vector<int>& meta_field_indices,
                    const std::vector<TimestampParser*>& timestamp_parsers,
                    bool strip_id = true,
                    bool strip_meta = true);
    
    std::shared_ptr<Record> call(const std::vector<std::string>& fields) override;

private:
    CheckFields* _is_other;
    int64_t _base_date;
    std::vector<int> _timestamp_indices;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    std::vector<TimestampParser*> _timestamp_parsers;
    bool _strip_id;
    bool _strip_meta;
};

class TextRecordParser : public RecordParser {
public:
    TextRecordParser(std::vector<RecordFieldParser*> parsers, bool exclusive = true, char separator = ',' );
    void add_line(const std::string& line) override;
    std::shared_ptr<Record> parse() override;
private:
    std::vector<RecordFieldParser*> _parsers;
    bool _exclusive;
    char _separator;
    std::vector<std::string> _headers;
    std::size_t _parse_index;
    std::vector<std::string> _fields;
};

#endif /* text_file_parsers_hpp */

