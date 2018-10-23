
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

int64_t FormatTimestampParser::call(const std::string& time) {
    ostringstream ostr;
    ostr << "1970-01-01T" << time;
    int64_t timestamp = str_to_timestamp(ostr.str(), _time_format, _micros);
    if (_micros) timestamp += _base_date * 1000;
    else timestamp += _base_date;
    return timestamp;
}

ParseException::ParseException(const char* m) : std::runtime_error(m) { }

TextQuoteParser::TextQuoteParser(CheckFields* is_quote,
                                int64_t base_date,
                                int timestamp_idx,
                                int bid_offer_idx,
                                int price_idx,
                                int qty_idx,
                                const vector<int>& id_field_indices,
                                const vector<int>& meta_field_indices,
                                TimestampParser* timestamp_parser,
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
_strip_meta(strip_meta) {
    if (!is_quote || !timestamp_parser) error("is_quote and timestamp parser must be specified")
}
    
shared_ptr<QuoteRecord> TextQuoteParser::call(const vector<string>& fields) {
    if (!_is_quote->call(fields)) return nullptr;
    int64_t timestamp = _timestamp_parser->call(fields[_timestamp_idx]) + _base_date;
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

TextTradeParser::TextTradeParser(CheckFields* is_trade,
                                 int64_t base_date,
                                int timestamp_idx,
                                int price_idx,
                                int qty_idx,
                                const vector<int>& id_field_indices,
                                const vector<int>& meta_field_indices,
                                TimestampParser* timestamp_parser,
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
_strip_meta(strip_meta) {
    if (!is_trade || !timestamp_parser) error("is_trade and timestamp parser must be specified")
}

shared_ptr<TradeRecord> TextTradeParser::call(const vector<string>& fields) {
    if (!_is_trade->call(fields)) return nullptr;
    int64_t timestamp = _timestamp_parser->call(fields[_timestamp_idx]) + _base_date;
    float price = str_to_float(fields[_price_idx]) / _price_multiplier;
    float qty = str_to_float(fields[_qty_idx]);
    string id = join_fields(fields, _id_field_indices, '|', _strip_id);
    string meta = join_fields(fields, _meta_field_indices, '|', _strip_meta);
    return std::make_shared<TradeRecord>(id, timestamp, qty, price, meta);
}
    
TextOpenInterestParser::TextOpenInterestParser(CheckFields* is_open_interest,
                                               int64_t base_date,
                                               int timestamp_idx,
                                               int qty_idx,
                                               const vector<int>& id_field_indices,
                                               const vector<int>& meta_field_indices,
                                               TimestampParser* timestamp_parser,
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
_strip_meta(strip_meta) {
    if (!is_open_interest || !timestamp_parser) error("is_open_interest and timestamp parser must be specified")
}

shared_ptr<OpenInterestRecord> TextOpenInterestParser::call(const vector<string>& fields) {
    if (!_is_open_interest->call(fields)) return nullptr;
    int64_t timestamp = _timestamp_parser->call(fields[_timestamp_idx]) + _base_date;
    float qty = str_to_float(fields[_qty_idx]);
    string id = join_fields(fields, _id_field_indices, '|', _strip_id);
    string meta = join_fields(fields, _meta_field_indices, '|', _strip_meta);
    return std::make_shared<OpenInterestRecord>(id, timestamp, qty, meta);
}
    
TextOtherParser::TextOtherParser(CheckFields* is_other,
                                int64_t base_date,
                                int timestamp_idx,
                                const vector<int>& id_field_indices,
                                const vector<int>& meta_field_indices,
                                TimestampParser* timestamp_parser,
                                bool strip_id,
                                bool strip_meta) :
_is_other(is_other),
_base_date(base_date),
_timestamp_idx(timestamp_idx),
_id_field_indices(id_field_indices),
_meta_field_indices(meta_field_indices),
_timestamp_parser(timestamp_parser),
_strip_id(strip_id),
_strip_meta(strip_meta) {
    if (!is_other || !timestamp_parser) error("is_other and timestamp parser must be specified")
}
    
shared_ptr<OtherRecord> TextOtherParser::call(const vector<string>& fields) {
    if (!_is_other->call(fields)) return nullptr;
    int64_t timestamp = _timestamp_parser->call(fields[_timestamp_idx]) + _base_date;
    string id = join_fields(fields, _id_field_indices, '|', _strip_id);
    string meta = join_fields(fields, _meta_field_indices, '|', _strip_meta);
    return shared_ptr<OtherRecord>(new OtherRecord(id, timestamp, meta));
}
    
TextRecordParser::TextRecordParser(TextQuoteParser* quote_parser,
                                   TextTradeParser* trade_parser,
                                   TextOpenInterestParser* open_interest_parser,
                                   TextOtherParser* other_parser,
                                   char separator) :
_quote_parser(quote_parser),
_trade_parser(trade_parser),
_open_interest_parser(open_interest_parser),
_other_parser(other_parser),
_separator(separator) {
    if (!((quote_parser) || trade_parser || open_interest_parser || other_parser)) error("at least one parser must be specified");
}
    
shared_ptr<Record> TextRecordParser::call(const std::string& line) {
    vector<string> fields = tokenize(line.c_str(), _separator);
    shared_ptr<Record> record = nullptr;
    if (_quote_parser) record = _quote_parser->call(fields);
    if (record) return record;
    if (_trade_parser) record = _trade_parser->call(fields);
    if (record) return record;
    if (_open_interest_parser) record = _open_interest_parser->call(fields);
    if (record) return record;
    if (_other_parser) record = _other_parser->call(fields);
    return record;
}

int64_t FastTimeMilliParser::call(const std::string& time) {
    if (time.size() != 12 || time[2] != ':' || time[5] != ':' || time[8] != '.') error("timestamp not in HH:MM:SS format: " << time)
        int hours = str_to_int(time.substr(0, 2).c_str());
    int minutes = str_to_int(time.substr(3, 2).c_str());
    int seconds = str_to_int(time.substr(6, 2).c_str());
    int millis = str_to_int(time.substr(9, 3).c_str());
    return static_cast<int64_t>((hours * 60 * 60 + minutes * 60 + seconds) * 1000 + millis);
}

int64_t FastTimeMicroParser::call(const std::string& time) {
    if (time.size() != 15 || time[2] != ':' || time[5] != ':' || time[8] != '.') error("timestamp not in HH:MM:SS format: " << time)
        int hours = str_to_int(time.substr(0, 2).c_str());
    int minutes = str_to_int(time.substr(3, 2).c_str());
    int seconds = str_to_int(time.substr(5, 2).c_str());
    int micros = str_to_int(time.substr(9, 6).c_str());
    return static_cast<int64_t>((hours * 60 * 60 + minutes * 60 + seconds) * 1000 + micros);
}
