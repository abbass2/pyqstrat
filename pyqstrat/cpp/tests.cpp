#include <chrono>
#include "arrow_writer.hpp"
#include "text_file_parsers.hpp"
#include "aggregators.hpp"
#include "text_file_processor.hpp"
#include "py_import_call_execute.hpp"

using namespace std::chrono;
using namespace std;

class IsTrade : public CheckFields {
public:
    IsTrade() : _is_field_in_list(make_shared<IsFieldInList>(2, vector<string>{"T"})) {}
    bool call(const vector<string>& fields) override { return _is_field_in_list->call(fields); }
private:
    shared_ptr<IsFieldInList> _is_field_in_list;
};

class IsQuote : public CheckFields {
public:
  IsQuote() : _is_field_in_list(make_shared<IsFieldInList>(2, vector<string>{"F", "N"})) {}
  bool call(const vector<string>& fields) override { return _is_field_in_list->call(fields); }
private:
    shared_ptr<IsFieldInList> _is_field_in_list;
};
                                
class IsOpenInterest : public CheckFields {
public:
    IsOpenInterest() : _is_field_in_list(make_shared<IsFieldInList>(2, vector<string>{"O"})) {}
    bool call(const vector<string>& fields) override { return _is_field_in_list->call(fields); }
private:
    shared_ptr<IsFieldInList> _is_field_in_list;
};

class IsOther : public CheckFields {
public:
    IsOther() : _is_field_in_list(make_shared<IsFieldInList>(2, vector<string>{"T", "F", "N", "O"})) {}
    bool call(const vector<string>& fields) override { return !_is_field_in_list->call(fields); }
private:
    shared_ptr<IsFieldInList> _is_field_in_list;
};

void test_tick_processing() {
    cout << "starting" << endl;
    int batch_size = 10000;
    //auto timestamp_parser = TimeOnlyTimestampParser("2018-01-15");
    auto timestamp_parser = FixedWidthTimeParser(false, 0, 2, 3, 2, 6, 2, 9, 3);
    auto is_quote = IsQuote();
    auto is_trade = IsTrade();
    auto is_open_interest = IsOpenInterest();
    auto is_other = IsOther();
    
    auto quote_parser = make_shared<TextQuoteParser>(&is_quote, 0, vector<int>{0}, 3, 9, 8, vector<int>{5, 6, 7}, vector<int>{10, 4}, vector<TimestampParser*>{&timestamp_parser}, "B", "O", 10000.0);
    auto trade_parser = make_shared<TextTradeParser>(&is_trade, 0, vector<int>{0}, 9, 8, vector<int>{5, 6, 7}, vector<int>{10, 4}, vector<TimestampParser*>{&timestamp_parser}, 10000.0);
    auto oi_parser = make_shared<TextOpenInterestParser>(&is_open_interest, 0, vector<int>{0}, 8, vector<int>{5, 6, 7}, vector<int>{10, 4}, vector<TimestampParser*>{&timestamp_parser});
    auto other_parser = make_shared<TextOtherParser>(&is_other, 0, vector<int>{0}, vector<int>{5, 6, 7}, vector<int>{10, 4}, vector<TimestampParser*>{&timestamp_parser});
    auto text_record_parser = TextRecordParser(std::vector<RecordFieldParser*>{quote_parser.get(), trade_parser.get(), oi_parser.get(), other_parser.get()});
    auto arrow_writer_creator = ArrowWriterCreator();
    //auto quote_aggregator = AllQuoteAggregator(writer_creator, "/tmp/quotes_all", batch_size);
    auto quote_aggregator = QuoteTOBAggregator(&arrow_writer_creator, "/tmp/quotes", "1m", true);
    //auto trade_aggregator = AllTradeAggregator(writer_creator, "/tmp/trades", batch_size);
    auto trade_aggregator = TradeBarAggregator(&arrow_writer_creator, "/tmp/trades", "1m", true);
    auto open_interest_aggregator = AllOpenInterestAggregator(&arrow_writer_creator, "/tmp/open_interest", batch_size);
    auto other_aggregator = AllOtherAggregator(&arrow_writer_creator, "/tmp/other", batch_size);
    std::vector<Aggregator*> aggregators = {&quote_aggregator, &trade_aggregator, &open_interest_aggregator, &other_aggregator};
    
    auto text_file_decompressor = TextFileDecompressor();
    auto substring_line_filter = SubStringLineFilter({",T,", ",F,", ",N,", ",O,", ",X,"});
    auto print_bad_line_handler = PrintBadLineHandler();
    auto price_qty_missing_data_handler = PriceQtyMissingDataHandler();
    auto processor = TextFileProcessor(&text_file_decompressor,
                                       &substring_line_filter,
                                       &text_record_parser,
                                       &print_bad_line_handler,
                                       nullptr,
                                       &price_qty_missing_data_handler,
                                       aggregators,
                                       1);
    auto start = high_resolution_clock::now();
    //int lines = processor.call("/Users/sal/Developer/coatbridge/vendor_data/algoseek/spx_dailies/w_2018-03-29.csv.gz", "gzip");
    int lines = processor.call("/tmp/BRKA_2018-01-01_data.gz", "gzip");

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    cout << "start: " << start.time_since_epoch().count() << " processed " << lines << " lines in " << duration.count() / 1000.0 << " milliseconds" << endl;
}

