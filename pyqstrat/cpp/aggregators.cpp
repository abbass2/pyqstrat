#include <algorithm>
#include "aggregators.hpp"

using namespace std;

int64_t parse_frequency(const string& frequency_str) {
    char unit = frequency_str[frequency_str.size() - 1];
    int64_t qty = std::stol(frequency_str.substr(0, frequency_str.size() - 1));
    int64_t multiplier = 0;
    switch (unit) {
        case 's': multiplier = 1000; break;
        case 'm': multiplier = 60 * 1000; break;
        case 'h': multiplier = 60 * 60 * 1000; break;
        case 'd': multiplier = 24 * 60 * 60 * 1000; break;
        default: error("unknown time unit: " << unit);
    }
    return qty * multiplier;
}

SymbolTradeBar::SymbolTradeBar (std::shared_ptr<Writer> writer, const std::string& id, int64_t frequency) :
    _writer(writer),
    _id(id),
    _frequency(frequency),
    _last_update(0),
    _o(NAN),
    _h(std::numeric_limits<float>::min()),
    _l(std::numeric_limits<float>::max()),
    _c(NAN),
    _v(0),
    _total_volume(0),
    _price_volume(0),
    _time_unit(0),
    _line_number(0),
    _written_line_number(0),
    _closed(false) {}

void SymbolTradeBar::write_records() {
    if (_records.empty()) return;
    for (auto record : _records) {
        _writer->add_record(record.first, *(record.second));
    }
}
    
shared_ptr<Tuple> SymbolTradeBar::get_curr_row() {
    if (_h == std::numeric_limits<float>::min()) _h = NAN;
    if (_l == std::numeric_limits<float>::max()) _l = NAN;
    shared_ptr<Tuple> tuple(new Tuple());
    tuple->add(_id);
    tuple->add((_time_unit + 1) * _frequency);
    tuple->add(_last_update);
    tuple->add(_o);
    tuple->add(_h);
    tuple->add(_l);
    tuple->add(_c);
    tuple->add(_v);
    tuple->add(_price_volume / _total_volume);
    return tuple;
}
    
void SymbolTradeBar::write_record(int line_number) {
    if (_line_number == 0) return; //Don't write out first record until bar is done
    _records.push_back(std::make_pair(line_number, get_curr_row()));
}
    
void SymbolTradeBar::init_bar(const TradeRecord& trade) {
    _last_update = trade.timestamp;
    _o = trade.price;
    _h = trade.price;
    _l = trade.price;
    _c = trade.price;
    _v = trade.qty;
    _price_volume = trade.price * trade.qty;
    _total_volume = trade.qty;
}
    
void SymbolTradeBar::update_bar(const TradeRecord& trade) {
    _id = trade.id;
    _last_update = trade.timestamp;
    _h = std::max(_h, trade.price);
    _l = std::min(_l, trade.price);
    _c = trade.price;
    _v = trade.qty;
    _price_volume += trade.price * trade.qty;
    _total_volume += trade.qty;
}
    
void SymbolTradeBar::add_trade(const TradeRecord& trade, int line_number) {
    if (std::isnan(trade.price) || (std::isnan(trade.qty))) return;
    int time_unit = static_cast<int>(trade.timestamp * 1.0 / _frequency);
    if (time_unit > _time_unit) { //We are into the next bar, write out results from previous bar
        write_record(_line_number);
        _time_unit = time_unit;
        init_bar(trade);
    } else {
        update_bar(trade);
    }
    _line_number = line_number;
}
    
void SymbolTradeBar::close() {
    if (_closed) return;
    if (_line_number > _written_line_number) { // write out last bar
        write_record(_line_number);
    }
    write_records();
    _closed = true;
}

SymbolTradeBar::~SymbolTradeBar() {
    close();
}

TradeBarAggregator::TradeBarAggregator(WriterCreator* writer_creator, const std::string& frequency, Schema::Type timestamp_unit) {
    if (!writer_creator) error("writer creator must be specified");
    if (frequency.empty()) error("frequency must be specified");
    _frequency = parse_frequency(frequency);
    Schema schema;
    schema.types = {
        std::make_pair("id", Schema::STRING),
        std::make_pair("timestamp", timestamp_unit),
        std::make_pair("last_update", timestamp_unit),
        std::make_pair("o", Schema::FLOAT32),
        std::make_pair("h", Schema::FLOAT32),
        std::make_pair("l", Schema::FLOAT32),
        std::make_pair("c", Schema::FLOAT32),
        std::make_pair("v", Schema::FLOAT32),
        std::make_pair("vwap", Schema::FLOAT32)
    };
    _writer = writer_creator->call("trade_bars", schema);
}
    
void TradeBarAggregator::call(const Record* record, int line_number) {
    if (record == nullptr) return;
    auto ptrade = dynamic_cast<const TradeRecord*>(record);
    if (ptrade == nullptr) return;
    auto trade = *ptrade;
    
    if (_trade_bars_by_symbol.find(trade.id) == _trade_bars_by_symbol.end()) {
        _trade_bars_by_symbol.insert(std::make_pair(trade.id, std::shared_ptr<SymbolTradeBar>(
            new SymbolTradeBar(_writer, trade.id, _frequency))));
    }
    std::shared_ptr<SymbolTradeBar> sym_trade_bar = _trade_bars_by_symbol.find(trade.id)->second;
    sym_trade_bar->add_trade(trade, line_number);
}
    
void TradeBarAggregator::close() {
    std::for_each(_trade_bars_by_symbol.begin(), _trade_bars_by_symbol.end(), [](auto& entry){ entry.second->close(); });
}

TradeBarAggregator::~TradeBarAggregator() {
    close();
}

SymbolQuoteTOB::SymbolQuoteTOB(std::shared_ptr<Writer> writer, const std::string& id, int64_t frequency) :
_writer(writer),
_id(id),
_timestamp(0),
_bid(NAN),
_ask(NAN),
_bid_size(NAN),
_ask_size(NAN),
_frequency(frequency),
_time_unit(0),
_line_number(0),
_written_line_number(0),
_closed(false ) {}

void SymbolQuoteTOB::write_records() {
    for (auto record : _records) {
        _writer->add_record(record.first, *record.second);
    }
}

shared_ptr<Tuple> SymbolQuoteTOB::get_curr_row() {
    shared_ptr<Tuple> tuple(new Tuple());
    tuple->add(_id);
    if (_frequency != -1) tuple->add((_time_unit + 1) * _frequency);
    tuple->add(_last_update);
    tuple->add(_bid);
    tuple->add(_bid_size);
    tuple->add(_ask);
    tuple->add(_ask_size);
    return tuple;
}

void SymbolQuoteTOB::write_record(int line_number) {
    if (_line_number == 0) return;  // Don't write out very first record
    _records.push_back(std::make_pair(line_number, get_curr_row()));
}

void SymbolQuoteTOB::update_row(const QuoteRecord& quote) {
    if (quote.bid) {
        _bid = quote.price;
        _bid_size = quote.qty;
    } else {
        _ask = quote.price;
        _ask_size = quote.qty;
    }
    _id = quote.id;
    _last_update = quote.timestamp;
}

void SymbolQuoteTOB::add_quote(const QuoteRecord& quote, int line_number) {
    if (_frequency != -1) {
        int time_unit = static_cast<int>(quote.timestamp * 1.0 / _frequency);
        if (time_unit > _time_unit) { //We are into the next bar, write out results from previous bar
            write_record(_line_number); //don't write out first row until first bar is done
            _time_unit = time_unit;
        }
    } else if (quote.timestamp != _timestamp) { // Only write out bid / ask when we have both, so wait for new record before writing out old one
        write_record(line_number);
    }

    update_row(quote);
    _line_number = line_number;
    _timestamp = quote.timestamp;
}

void SymbolQuoteTOB::close() {
    if (_closed) return;
    if (_line_number > _written_line_number) { // write out last bar
        write_record(_line_number);
    }
    write_records();
    _closed = true;
}

SymbolQuoteTOB::~SymbolQuoteTOB() {
    close();
}
    

//Assumes quotes are processed in time order.  Set frequency to "" to create bid / offer every time TOB changes.
QuoteTOBAggregator::QuoteTOBAggregator(WriterCreator* writer_creator,
                                       const std::string& frequency,
                                       Schema::Type timestamp_unit) :
        _frequency(-1) {
    if (frequency.size()) _frequency = parse_frequency(frequency);
    Schema schema;
    schema.types.push_back(std::make_pair("id", Schema::STRING));
    schema.types.push_back(std::make_pair("timestamp", timestamp_unit));
    if (frequency.size()) schema.types.push_back(std::make_pair("last_update", timestamp_unit));
    schema.types.push_back(std::make_pair("bid", Schema::FLOAT32));
    schema.types.push_back(std::make_pair("bid_size", Schema::FLOAT32));
    schema.types.push_back(std::make_pair("ask", Schema::FLOAT32));
    schema.types.push_back(std::make_pair("ask_size", Schema::FLOAT32));
    _writer = writer_creator->call("quote_tob", schema);
}

void QuoteTOBAggregator::call(const Record* record, int line_number) {
    if (record == nullptr) return;
    auto pquote = dynamic_cast<const QuoteRecord*>(record);
    if (pquote == nullptr) return;
    auto quote = *pquote;

    if (_tob_by_symbol.find(quote.id) == _tob_by_symbol.end()) {
        _tob_by_symbol.insert(std::make_pair(quote.id, std::shared_ptr<SymbolQuoteTOB>(
            new SymbolQuoteTOB(_writer, quote.id, _frequency))));
    }
    auto& tob = _tob_by_symbol.find(quote.id)->second;
    tob->add_quote(quote, line_number);
}

void QuoteTOBAggregator::close() {
    std::for_each(_tob_by_symbol.begin(), _tob_by_symbol.end(), [](auto& entry){ entry.second->close(); });
}

QuoteTOBAggregator::~QuoteTOBAggregator() {
    close();
}

AllQuoteAggregator::AllQuoteAggregator(WriterCreator* writer_creator, Schema::Type timestamp_unit) :
_line_number_offset(0) {
    if (!writer_creator) error("writer creator must be specified");
    Schema schema;
    schema.types = {
        std::make_pair("id", Schema::STRING),
        std::make_pair("timestamp", timestamp_unit),
        std::make_pair("bid", Schema::BOOL),
        std::make_pair("qty", Schema::FLOAT32),
        std::make_pair("price", Schema::FLOAT32),
        std::make_pair("meta", Schema::STRING),
    };
    _writer = writer_creator->call("quotes", schema);
}

void AllQuoteAggregator::call(const Record* record, int line_number) {
    if (record == nullptr) return;
    auto pquote = dynamic_cast<const QuoteRecord*>(record);
    if (pquote == nullptr) return;
    auto quote = *pquote;
    Tuple tuple;
    tuple.add(quote.id);
    tuple.add(quote.timestamp);
    tuple.add(quote.bid);
    tuple.add(quote.qty);
    tuple.add(quote.price);
    tuple.add(quote.metadata);
    _writer->add_record(line_number - _line_number_offset + 1, tuple);
}

AllQuotePairAggregator::AllQuotePairAggregator(WriterCreator* writer_creator, Schema::Type timestamp_unit):
_writer_creator(writer_creator) {
    _schema.types = {
            std::make_pair("id", Schema::STRING),
            std::make_pair("timestamp", timestamp_unit),
            std::make_pair("bid_price", Schema::FLOAT32),
            std::make_pair("bid_qty", Schema::FLOAT32),
            std::make_pair("ask_price", Schema::FLOAT32),
            std::make_pair("ask_qty", Schema::FLOAT32),
            std::make_pair("meta", Schema::STRING),
        };
}

void AllQuotePairAggregator::call(const Record* record, int line_number) {
    if (record == nullptr) return;
    auto pquote = dynamic_cast<const QuotePairRecord*>(record);
    if (pquote == nullptr) return;
    auto quote = *pquote;
    
    auto pair = _writers.find(quote.id);
    std::shared_ptr<Writer> writer;
    if (pair == _writers.end()) {
        writer = _writer_creator->call(quote.id, _schema);
        _writers.insert(make_pair(quote.id, writer));
    } else {
        writer = pair->second;
    }
    Tuple tuple;
    tuple.add(quote.id);
    tuple.add(quote.timestamp);
    tuple.add(quote.bid_price);
    tuple.add(quote.bid_qty);
    tuple.add(quote.ask_price);
    tuple.add(quote.ask_qty);
    tuple.add(quote.metadata);
    writer->add_record(line_number, tuple);
}

AllTradeAggregator::AllTradeAggregator(WriterCreator* writer_creator, Schema::Type timestamp_unit):
_writer_creator(writer_creator) {
    _schema.types = {
        std::make_pair("id", Schema::STRING),
        std::make_pair("timestamp", timestamp_unit),
        std::make_pair("qty", Schema::FLOAT32),
        std::make_pair("price", Schema::FLOAT32),
        std::make_pair("meta", Schema::STRING),
    };
}

void AllTradeAggregator::call(const Record* record, int line_number) {
    if (record == nullptr) return;
    auto ptrade = dynamic_cast<const TradeRecord*>(record);
    if (ptrade == nullptr) return;
    auto trade = *ptrade;
    
    auto pair = _writers.find(trade.id);
    std::shared_ptr<Writer> writer;
    if (pair == _writers.end()) {
        writer = _writer_creator->call(trade.id, _schema);
        _writers.insert(make_pair(trade.id, writer));
    } else {
        writer = pair->second;
    }
    
    Tuple tuple;
    tuple.add(trade.id);
    tuple.add(trade.timestamp);
    tuple.add(trade.qty);
    tuple.add(trade.price);
    tuple.add(trade.metadata);
    writer->add_record(line_number, tuple);
}

AllOpenInterestAggregator::AllOpenInterestAggregator(WriterCreator* writer_creator, Schema::Type timestamp_unit) {
    if (!writer_creator) error("writer creator must be specified");
    Schema schema;
    schema.types = {
        std::make_pair("id", Schema::STRING),
        std::make_pair("timestamp", timestamp_unit),
        std::make_pair("qty", Schema::FLOAT32),
        std::make_pair("meta", Schema::STRING),
    };
    _writer = writer_creator->call("open_interest", schema);
}

void AllOpenInterestAggregator::call(const Record* record, int line_number) {
    if (record == nullptr) return;
    auto poi = dynamic_cast<const OpenInterestRecord*>(record);
    if (poi == nullptr) return;
    auto oi = *poi;

    Tuple tuple;
    tuple.add(oi.id);
    tuple.add(oi.timestamp);
    tuple.add(oi.qty);
    tuple.add(oi.metadata);
    _writer->add_record(line_number, tuple);
}

AllOtherAggregator::AllOtherAggregator(WriterCreator* writer_creator, Schema::Type timestamp_unit) {
    if (!writer_creator) error("writer creator must be specified");
    Schema schema;
    schema.types = {
        std::make_pair("id", Schema::STRING),
        std::make_pair("timestamp", timestamp_unit),
        std::make_pair("meta", Schema::STRING),
    };
    _writer = writer_creator->call("other", schema);
}

void AllOtherAggregator::call(const Record* record, int line_number) {
    if (record == nullptr) return;
    auto pother = dynamic_cast<const OtherRecord*>(record);
    if (pother == nullptr) return;
    auto other = *pother;

    Tuple tuple;
    tuple.add(other.id);
    tuple.add(other.timestamp);
    tuple.add(other.metadata);
    _writer->add_record(line_number, tuple);
}

