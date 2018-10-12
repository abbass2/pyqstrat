#ifndef aggregators_hpp
#define aggregators_hpp

#include <string>
#include <map>
#include <cmath>
#include <tuple>

#include "utils.hpp"
#include "types.hpp"

int64_t parse_frequency(const std::string& frequency_str);

typedef std::function<std::shared_ptr<Writer>(const std::string&, const Schema&, bool, int)> WriterCreator;

class SymbolTradeBar final {
public:
    SymbolTradeBar (std::shared_ptr<Writer>, const std::string& id, bool batch_by_id, int64_t frequency);
    void add_trade(const TradeRecord& trade, int line_number);
    void close();
    virtual ~SymbolTradeBar();
private:
    void write_records();
    void write_record(int line_number);
    void init_bar(const TradeRecord& trade);
    void update_bar(const TradeRecord& trade);
    std::shared_ptr<Tuple> get_curr_row();
    
    std::shared_ptr<Writer> _writer;
    std::string _id;
    bool _batch_by_id;
    int64_t _frequency;
    int64_t _last_update;
    float _o;
    float _h;
    float _l;
    float _c;
    float _v;
    float _total_volume;
    float _price_volume;
    int _time_unit;
    int _line_number;
    int _written_line_number;
    std::vector<std::pair<int, std::shared_ptr<Tuple>>> _records;
    bool _closed;
};

class TradeBarAggregator final {
public:
    TradeBarAggregator(WriterCreator, const std::string& output_file_prefix, const std::string& frequency = "5m",
                       bool batch_by_id = true, int batch_size = std::numeric_limits<int>::max(),
                       Schema::Type timestamp_unit = Schema::TIMESTAMP_MILLI);
    void operator()(const TradeRecord& trade, int line_number);
    void close();
    ~TradeBarAggregator();
private:
    std::shared_ptr<Writer> _writer;
    bool _batch_by_id;
    int _batch_size;
    int64_t _frequency;
    std::map<std::string, std::shared_ptr<SymbolTradeBar>> _trade_bars_by_symbol;
    int _record_num;
};

class SymbolQuoteTOB final {
public:
    SymbolQuoteTOB(std::shared_ptr<Writer>, const std::string& id, bool batch_by_id, int64_t frequency);
    void add_quote(const QuoteRecord& quote, int line_number);
    void close();
    ~SymbolQuoteTOB();
private:
    std::shared_ptr<Tuple> get_curr_row();
    void write_record(int line_number);
    void update_row(const QuoteRecord& trade);
    void write_records();

    std::shared_ptr<Writer> _writer;
    std::string _id;
    bool _batch_by_id;
    int64_t _timestamp;
    int64_t _last_update;
    float _bid;
    float _ask;
    float _bid_size;
    float _ask_size;
    int64_t _frequency;
    int _time_unit;
    int _line_number;
    int _written_line_number;
    std::vector<std::pair<int, std::shared_ptr<Tuple>>> _records;
    bool _closed;
};

class QuoteTOBAggregator final {
public:
    //Assumes quotes are processed in time order.  Set frequency to "" to create bid / offer every time TOB changes.
    QuoteTOBAggregator(WriterCreator, const std::string& output_file_prefix, const std::string& frequency = "5m",
                       bool batch_by_id = true, int batch_size = std::numeric_limits<int>::max(),
                       Schema::Type timestamp_unit = Schema::TIMESTAMP_MILLI);
    void operator()(const QuoteRecord& quote, int line_number);
    void close();
    virtual ~QuoteTOBAggregator();
private:
    std::shared_ptr<Writer> _writer;
    bool _batch_by_id;
    int _batch_size;
    int64_t _frequency;
    int _record_num;
    std::map<std::string, std::shared_ptr<SymbolQuoteTOB>> _tob_by_symbol;
};

class AllQuoteAggregator final {
public:
    AllQuoteAggregator(WriterCreator, const std::string& output_file_prefix,
                       int batch_size = 10000, Schema::Type timestamp_unit = Schema::TIMESTAMP_MILLI);
    void operator()(const QuoteRecord& quote, int line_number);
private:
    std::shared_ptr<Writer> _writer;
    std::string _id;
};

class AllTradeAggregator final {
public:
    AllTradeAggregator(WriterCreator, const std::string& output_file_prefix, int batch_size = 10000,
                       Schema::Type timestamp_unit = Schema::TIMESTAMP_MILLI);
    void operator()(const TradeRecord& trade, int line_number);
private:
    std::shared_ptr<Writer> _writer;
    std::string _id;
};

class AllOpenInterestAggregator final {
public:
    AllOpenInterestAggregator(WriterCreator, const std::string& output_file_prefix,int batch_size = 10000,
                              Schema::Type timestamp_unit = Schema::TIMESTAMP_MILLI);
    void operator()(const OpenInterestRecord& oi, int line_number);
private:
    std::shared_ptr<Writer> _writer;
    std::string _id;
};

class AllOtherAggregator final {
public:
    AllOtherAggregator(WriterCreator, const std::string& output_file_prefix, int batch_size = 10000,
                       Schema::Type timestamp_unit = Schema::TIMESTAMP_MILLI);
    void operator()(const OtherRecord& other, int line_number);
private:
    std::shared_ptr<Writer> _writer;
    std::string _id;
};

#endif
