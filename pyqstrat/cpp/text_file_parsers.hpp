#ifndef text_file_parsers_hpp
#define text_file_parsers_hpp

#include <string>
#include <vector>

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

struct FastTimeMilliParser : public TimestampParser {
    int64_t call(const std::string& time) override;
};

struct FastTimeMicroParser : public TimestampParser {
    int64_t call(const std::string& time) override;
};

class TextQuoteParser : public QuoteParser {
public:
    TextQuoteParser(CheckFields* is_quote,
                    int64_t base_date,
                    int timestamp_idx,
                    int bid_offer_idx,
                    int price_idx,
                    int qty_idx,
                    const std::vector<int>& id_field_indices,
                    const std::vector<int>& meta_field_indices,
                    TimestampParser* timestamp_parser,
                    const std::string& bid_str,
                    const std::string& offer_str,
                    float price_multiplier = 1.0,
                    bool strip_id = true,
                    bool strip_meta = true);
    
    std::shared_ptr<QuoteRecord> call(const std::vector<std::string>& fields) override;
private:
    CheckFields* _is_quote;
    int64_t _base_date;
    int _timestamp_idx;
    int _bid_offer_idx;
    int _price_idx;
    int _qty_idx;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    TimestampParser* _timestamp_parser;
    std::string _bid_str;
    std::string _offer_str;
    float _price_multiplier;
    bool _strip_id;
    bool _strip_meta;
};

class TextTradeParser : public TradeParser {
public:
    TextTradeParser(CheckFields* is_trade,
                    int64_t base_date,
                    int timestamp_idx,
                    int price_idx,
                    int qty_idx,
                    const std::vector<int>& id_field_indices,
                    const std::vector<int>& meta_field_indices,
                    TimestampParser* timestamp_parser,
                    float price_multiplier = 1,
                    bool strip_id = true,
                    bool strip_meta = true);
    
    std::shared_ptr<TradeRecord> call(const std::vector<std::string>& fields) override;
    
private:
    CheckFields* _is_trade;
    int64_t _base_date;
    int _timestamp_idx;
    int _price_idx;
    int _qty_idx;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    TimestampParser* _timestamp_parser;
    float _price_multiplier;
    bool _strip_id;
    bool _strip_meta;
};

class TextOpenInterestParser : public OpenInterestParser {
public:
    TextOpenInterestParser(CheckFields* is_open_interest,
                           int64_t base_date,
                           int timestamp_idx,
                           int qty_idx,
                           const std::vector<int>& id_field_indices,
                           const std::vector<int>& meta_field_indices,
                           TimestampParser* timestamp_parser,
                           bool strip_id = true,
                           bool strip_meta = true);
    
    std::shared_ptr<OpenInterestRecord> call(const std::vector<std::string>& fields) override;
    
private:
    CheckFields* _is_open_interest;
    int64_t _base_date;
    int _timestamp_idx;
    int _qty_idx;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    TimestampParser* _timestamp_parser;
    bool _strip_id;
    bool _strip_meta;
};

class TextOtherParser : public OtherParser {
public:
    TextOtherParser(CheckFields* is_other,
                    int64_t base_date,
                    int timestamp_idx,
                    const std::vector<int>& id_field_indices,
                    const std::vector<int>& meta_field_indices,
                    TimestampParser* timestamp_parser,
                    bool strip_id = true,
                    bool strip_meta = true);
    
    std::shared_ptr<OtherRecord> call(const std::vector<std::string>& fields) override;

private:
    CheckFields* _is_other;
    int64_t _base_date;
    int _timestamp_idx;
    std::vector<int> _id_field_indices;
    std::vector<int> _meta_field_indices;
    TimestampParser* _timestamp_parser;
    bool _strip_id;
    bool _strip_meta;
};

class TextRecordParser : public RecordParser {
public:
    
    TextRecordParser(TextQuoteParser* quote_parser, TextTradeParser* trade_parser,
                     TextOpenInterestParser* open_interest_parser,
                     TextOtherParser* other_parser,
                     char separator = ',' );
    
    std::shared_ptr<Record> call(const std::string& line) override;
    
private:
    TextQuoteParser* _quote_parser;
    TextTradeParser* _trade_parser;
    TextOpenInterestParser* _open_interest_parser;
    TextOtherParser* _other_parser;
    std::vector<std::string> _headers;
    char _separator;
};

#endif /* text_file_parsers_hpp */

