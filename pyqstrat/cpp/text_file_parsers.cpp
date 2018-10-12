
#include "text_file_parsers.hpp"

using namespace std;

#define parse_error(msg) \
{ \
std::ostringstream os; \
os << msg << " file: " << __FILE__ << "line: " << __LINE__ ; \
throw ParseException(os.str().c_str()); \
}

FormatTimestampParser::FormatTimestampParser(int64_t base_date, const std::string& time_format, bool micros) :
_base_date(base_date),
_micros(micros) {
    _time_format = "%Y-%m-%dT" + time_format;
}

int64_t FormatTimestampParser::operator()(const std::string& time) {
    ostringstream ostr;
    ostr << "1970-01-01T" << time;
    int64_t timestamp = str_to_timestamp(ostr.str(), _time_format, _micros);
    if (_micros) timestamp += _base_date * 1000;
    else timestamp += _base_date;
    return timestamp;
}

ParseException::ParseException(const char* m) : std::runtime_error(m) { }

TextQuoteParser::TextQuoteParser(IsRecordFunc is_quote,
                                int64_t base_date,
                                int timestamp_idx,
                                int bid_offer_idx,
                                int price_idx,
                                int qty_idx,
                                const vector<int>& id_field_indices,
                                const vector<int>& meta_field_indices,
                                TimestampParser timestamp_parser,
                                const string& bid_str,
                                const string& offer_str,
                                float price_multiplier,
                                bool strip_id,
                                bool strip_meta) :
_is_quote(is_quote),
_base_date(base_date),
_timestamp_idx(timestamp_idx),
_bid_offer_idx(bid_offer_idx),
_price_idx(price_idx),
_qty_idx(qty_idx),
_id_field_indices(id_field_indices),
_meta_field_indices(meta_field_indices),
_timestamp_parser(timestamp_parser),
_bid_str(bid_str),
_offer_str(offer_str),
_price_multiplier(price_multiplier),
_strip_id(strip_id),
_strip_meta(strip_meta) {}
    
shared_ptr<QuoteRecord> TextQuoteParser::operator()(const vector<string>& fields) {
    if (!_is_quote(fields)) return nullptr;
    int64_t timestamp = _timestamp_parser(fields[_timestamp_idx]) + _base_date;
    const string& bid_offer_str = fields[_bid_offer_idx];
    bool bid;
    if (bid_offer_str == _bid_str) bid = true;
    else if (bid_offer_str == _offer_str) bid = false;
    else parse_error("unknown bid offer string: " << bid_offer_str);
    float price = str_to_float(fields[_price_idx]) / _price_multiplier;
    float qty = str_to_float(fields[_qty_idx]);
    string id = join_fields(fields, _id_field_indices, '|', _strip_id);
    string meta = join_fields(fields, _meta_field_indices, '|', _strip_meta);
    return shared_ptr<QuoteRecord>(new QuoteRecord(id, timestamp, bid, qty, price, meta));
}

TextTradeParser::TextTradeParser(IsRecordFunc is_trade,
                                 int64_t base_date,
                                int timestamp_idx,
                                int price_idx,
                                int qty_idx,
                                const vector<int>& id_field_indices,
                                const vector<int>& meta_field_indices,
                                TimestampParser timestamp_parser,
                                float price_multiplier,
                                bool strip_id,
                                bool strip_meta) :
_is_trade(is_trade),
_base_date(base_date),
_timestamp_idx(timestamp_idx),
_price_idx(price_idx),
_qty_idx(qty_idx),
_id_field_indices(id_field_indices),
_meta_field_indices(meta_field_indices),
_timestamp_parser(timestamp_parser),
_price_multiplier(price_multiplier),
_strip_id(strip_id),
_strip_meta(strip_meta) {}

shared_ptr<TradeRecord> TextTradeParser::operator()(const vector<string>& fields) {
    if (!_is_trade(fields)) return nullptr;
    int64_t timestamp = _timestamp_parser(fields[_timestamp_idx]) + _base_date;
    float price = str_to_float(fields[_price_idx]) / _price_multiplier;
    float qty = str_to_float(fields[_qty_idx]);
    string id = join_fields(fields, _id_field_indices, '|', _strip_id);
    string meta = join_fields(fields, _meta_field_indices, '|', _strip_meta);
    return std::make_shared<TradeRecord>(id, timestamp, qty, price, meta);
}
    
TextOpenInterestParser::TextOpenInterestParser(IsRecordFunc is_open_interest,
                                               int64_t base_date,
                                               int timestamp_idx,
                                               int qty_idx,
                                               const vector<int>& id_field_indices,
                                               const vector<int>& meta_field_indices,
                                               TimestampParser timestamp_parser,
                                               bool strip_id,
                                               bool strip_meta) :
_is_open_interest(is_open_interest),
_base_date(base_date),
_timestamp_idx(timestamp_idx),
_qty_idx(qty_idx),
_id_field_indices(id_field_indices),
_meta_field_indices(meta_field_indices),
_timestamp_parser(timestamp_parser),
_strip_id(strip_id),
_strip_meta(strip_meta) {}

shared_ptr<OpenInterestRecord> TextOpenInterestParser::operator()(const vector<string>& fields) {
    if (!_is_open_interest(fields)) return nullptr;
    int64_t timestamp = _timestamp_parser(fields[_timestamp_idx]) + _base_date;
    float qty = str_to_float(fields[_qty_idx]);
    string id = join_fields(fields, _id_field_indices, '|', _strip_id);
    string meta = join_fields(fields, _meta_field_indices, '|', _strip_meta);
    return std::make_shared<OpenInterestRecord>(id, timestamp, qty, meta);
}
    
TextOtherParser::TextOtherParser(IsRecordFunc is_other,
                                int64_t base_date,
                                int timestamp_idx,
                                const vector<int>& id_field_indices,
                                const vector<int>& meta_field_indices,
                                TimestampParser timestamp_parser,
                                bool strip_id,
                                bool strip_meta) :
_is_other(is_other),
_base_date(base_date),
_timestamp_idx(timestamp_idx),
_id_field_indices(id_field_indices),
_meta_field_indices(meta_field_indices),
_timestamp_parser(timestamp_parser),
_strip_id(strip_id),
_strip_meta(strip_meta) {}
    
shared_ptr<OtherRecord> TextOtherParser::operator()(const vector<string>& fields) {
    if (!_is_other(fields)) return nullptr;
    int64_t timestamp = _timestamp_parser(fields[_timestamp_idx]) + _base_date;
    string id = join_fields(fields, _id_field_indices, '|', _strip_id);
    string meta = join_fields(fields, _meta_field_indices, '|', _strip_meta);
    return shared_ptr<OtherRecord>(new OtherRecord(id, timestamp, meta));
}
    
TextRecordParser::TextRecordParser(shared_ptr<TextQuoteParser> quote_parser,
                                   shared_ptr<TextTradeParser> trade_parser,
                                   shared_ptr<TextOpenInterestParser> open_interest_parser,
                                   shared_ptr<TextOtherParser> other_parser,
                                   char separator) :
_quote_parser(quote_parser),
_trade_parser(trade_parser),
_open_interest_parser(open_interest_parser),
_other_parser(other_parser),
_separator(separator) {
    if (!((quote_parser) || trade_parser || open_interest_parser || other_parser)) error("at least one parser must be specified");
}
    
shared_ptr<Record> TextRecordParser::operator()(const std::string& line) {
    vector<string> fields = tokenize(line.c_str(), _separator);
    shared_ptr<Record> record = nullptr;
    if (_quote_parser) record = (*_quote_parser)(fields);
    if (record) return record;
    if (_trade_parser) record = (*_trade_parser)(fields);
    if (record) return record;
    if (_open_interest_parser) record = (*_open_interest_parser)(fields);
    if (record) return record;
    if (_other_parser) record = (*_other_parser)(fields);
    return record;
}
