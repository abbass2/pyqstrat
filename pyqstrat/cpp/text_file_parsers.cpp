
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

shared_ptr<Record> TextQuoteParser::call(const vector<string>& fields) {
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
    return shared_ptr<Record>(new QuoteRecord(id, timestamp, bid, qty, price, meta));
}


TextQuotePairParser::TextQuotePairParser(CheckFields* is_quote_pair,
                                 int64_t base_date,
                                 int timestamp_idx,
                                 int bid_price_idx,
                                 int bid_qty_idx,
                                 int ask_price_idx,
                                 int ask_qty_idx,
                                 const vector<int>& id_field_indices,
                                 const vector<int>& meta_field_indices,
                                 TimestampParser* timestamp_parser,
                                 float price_multiplier,
                                 bool strip_id,
                                 bool strip_meta) :
_is_quote_pair(is_quote_pair),
_base_date(base_date),
_timestamp_idx(timestamp_idx),
_bid_price_idx(bid_price_idx),
_bid_qty_idx(bid_qty_idx),
_ask_price_idx(ask_price_idx),
_ask_qty_idx(ask_qty_idx),
_id_field_indices(id_field_indices),
_meta_field_indices(meta_field_indices),
_timestamp_parser(timestamp_parser),
_price_multiplier(price_multiplier),
_strip_id(strip_id),
_strip_meta(strip_meta) {
    if (!timestamp_parser) error("timestamp parser must be specified")
}

shared_ptr<Record> TextQuotePairParser::call(const vector<string>& fields) {
    if (_is_quote_pair && !_is_quote_pair->call(fields)) return nullptr;
    int64_t timestamp = _timestamp_parser->call(fields[_timestamp_idx]) + _base_date;
    float bid_price = str_to_float(fields[_bid_price_idx]) / _price_multiplier;
    float bid_qty = str_to_float(fields[_bid_qty_idx]);
    float ask_price = str_to_float(fields[_ask_price_idx]) / _price_multiplier;
    float ask_qty = str_to_float(fields[_ask_qty_idx]);
    string id = join_fields(fields, _id_field_indices, '|', _strip_id);
    string meta = join_fields(fields, _meta_field_indices, '|', _strip_meta);
    return shared_ptr<Record>(new QuotePairRecord(id, timestamp, bid_price, bid_qty, ask_price, ask_qty, meta));
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

shared_ptr<Record> TextTradeParser::call(const vector<string>& fields) {
    if (!_is_trade->call(fields)) return nullptr;
    int64_t timestamp = _timestamp_parser->call(fields[_timestamp_idx]) + _base_date;
    float price = str_to_float(fields[_price_idx]) / _price_multiplier;
    float qty = str_to_float(fields[_qty_idx]);
    string id = join_fields(fields, _id_field_indices, '|', _strip_id);
    string meta = join_fields(fields, _meta_field_indices, '|', _strip_meta);
    return shared_ptr<Record>(new TradeRecord(id, timestamp, qty, price, meta));
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

shared_ptr<Record> TextOpenInterestParser::call(const vector<string>& fields) {
    if (!_is_open_interest->call(fields)) return nullptr;
    int64_t timestamp = _timestamp_parser->call(fields[_timestamp_idx]) + _base_date;
    float qty = str_to_float(fields[_qty_idx]);
    string id = join_fields(fields, _id_field_indices, '|', _strip_id);
    string meta = join_fields(fields, _meta_field_indices, '|', _strip_meta);
    return shared_ptr<Record>(new OpenInterestRecord(id, timestamp, qty, meta));
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
    
shared_ptr<Record> TextOtherParser::call(const vector<string>& fields) {
    if (!_is_other->call(fields)) return nullptr;
    int64_t timestamp = _timestamp_parser->call(fields[_timestamp_idx]) + _base_date;
    string id = join_fields(fields, _id_field_indices, '|', _strip_id);
    string meta = join_fields(fields, _meta_field_indices, '|', _strip_meta);
    return shared_ptr<Record>(new OtherRecord(id, timestamp, meta));
}
    
TextRecordParser::TextRecordParser(std::vector<RecordFieldParser*> parsers, bool exclusive, char separator) :
    _parsers(parsers),
    _exclusive(exclusive),
    _separator(separator),
    _parse_index(0) {
    if (parsers.empty()) error("at least one parser must be specified");
}

void TextRecordParser::add_line(const std::string& line) {
    _fields = tokenize(line.c_str(), _separator);
    _parse_index = 0;
}
    
shared_ptr<Record> TextRecordParser::parse() {
    shared_ptr<Record> record;
    for (;;) {
        _parse_index++;  // Make sure this gets incremented even if next line throws
        if (_parse_index == (_parsers.size() + 1)) break;
        record = _parsers[_parse_index - 1]->call(_fields);
        if (record) {
            //If exclusive don't try any parsers for the line after the first one that succeeds
            if (_exclusive) _parse_index = static_cast<int>(_parsers.size());
            return record;
        }
    }
    return record;
}

int get_time_part(const std::string& time, int start, int size) {
    if (start < 0 || size <= 0) {
        return 0;
    }
    int ret = str_to_int(time.substr(start, size).c_str());
    return ret;
}


FixedWidthTimeParser::FixedWidthTimeParser(bool micros,
                                 int hours_start,
                                 int hours_size,
                                 int minutes_start,
                                 int minutes_size,
                                 int seconds_start,
                                 int seconds_size,
                                 int millis_start,
                                 int millis_size,
                                 int micros_start,
                                 int micros_size) :
                                    _micros(micros),
                                    _hours_start(hours_start),
                                    _hours_size(hours_size),
                                    _minutes_start(minutes_start),
                                    _minutes_size(minutes_size),
                                    _seconds_start(seconds_start),
                                    _seconds_size(seconds_size),
                                    _millis_start(millis_start),
                                    _millis_size(millis_size),
                                    _micros_start(micros_start),
                                    _micros_size(micros_size)
{}


int64_t FixedWidthTimeParser::call(const std::string& time) {
    int hours = get_time_part(time, _hours_start, _hours_size);
    int minutes = get_time_part(time, _minutes_start, _minutes_size);
    int seconds = get_time_part(time, _seconds_start, _seconds_size);
    int millis = get_time_part(time, _millis_start, _millis_size);
    int micros = get_time_part(time, _micros_start, _micros_size);
    if (_micros) return static_cast<int64_t>((hours * 60 * 60 + minutes * 60 + seconds) * 1000000 + millis * 1000 + micros);
    return static_cast<int64_t>((hours * 60 * 60 + minutes * 60 + seconds) * 1000 + millis);
}

