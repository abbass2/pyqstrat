#include <chrono>
#include "arrow_writer.hpp"
#include "text_file_parsers.hpp"
#include "aggregators.hpp"
#include "text_file_processor.hpp"

using namespace std::chrono;
using namespace std;

shared_ptr<Writer> writer_creator(const std::string& output_file_prefix, const Schema& schema, bool create_batch_id_file, int batch_size) {
    return shared_ptr<Writer>(new ArrowWriter(output_file_prefix, schema, create_batch_id_file, batch_size));
}

bool is_trade(const std::vector<std::string>& fields) { return is_field_in_list(fields, 2, {"T"}); }
bool is_quote(const std::vector<std::string>& fields) { return is_field_in_list(fields, 2,  {"F", "N"}); }
bool is_open_interest(const std::vector<std::string>& fields) { return is_field_in_list(fields, 2,  {"O"}); }
bool is_other(const std::vector<std::string>& fields) { return !(is_field_in_list(fields, 2, {"T", "F", "N", "O"})); }


void test_tick_processing() {
    cout << "starting" << endl;
    int batch_size = 10000;
    //auto timestamp_parser = TimeOnlyTimestampParser("2018-01-15");
    auto timestamp_parser = fast_time_milli_parser;
    auto quote_parser = make_shared<TextQuoteParser>(is_quote, 0, 0, 3, 9, 8, vector<int>{5, 6, 7}, vector<int>{10, 4}, timestamp_parser, "B", "O", 10000.0);
    auto trade_parser = make_shared<TextTradeParser>(is_trade, 0, 0, 9, 8, vector<int>{5, 6, 7}, vector<int>{10, 4}, timestamp_parser, 10000.0);
    auto oi_parser = make_shared<TextOpenInterestParser>(is_open_interest, 0, 0, 8, vector<int>{5, 6, 7}, vector<int>{10, 4}, timestamp_parser);
    auto other_parser = make_shared<TextOtherParser>(is_other, 0, 0, vector<int>{5, 6, 7}, vector<int>{10, 4}, timestamp_parser);
    auto text_record_parser = TextRecordParser(quote_parser, trade_parser, oi_parser, other_parser);
    //auto quote_aggregator = AllQuoteAggregator(writer_creator, "/tmp/quotes_all", batch_size);
    auto quote_aggregator = QuoteTOBAggregator(writer_creator, "/tmp/quotes", "1m", true);
    //auto trade_aggregator = AllTradeAggregator(writer_creator, "/tmp/trades", batch_size);
    auto trade_aggregator = TradeBarAggregator(writer_creator, "/tmp/trades", "1m", true);
    auto open_interest_aggregator = AllOpenInterestAggregator(writer_creator, "/tmp/open_interest", batch_size);
    auto other_aggregator = AllOtherAggregator(writer_creator, "/tmp/other", batch_size);
    auto processor = TextFileProcessor(text_file_decompressor,
                                       SubStringLineFilter({",T,", ",F,", ",N,", ",O,", ",X,"}),
                                       text_record_parser,
                                       PrintBadLineHandler(),
                                       {},
                                       price_qty_missing_data_handler,
                                       quote_aggregator,
                                       trade_aggregator,
                                       open_interest_aggregator,
                                       other_aggregator,
                                       1);
    auto start = high_resolution_clock::now();
    //int lines = processor("/Users/sal/Developer/coatbridge/vendor_data/algoseek/spx_dailies/w_2018-03-29.csv.gz", "gzip");
    int lines = processor("/tmp/BRKA_2018-01-01_data.gz", "gzip");

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    cout << "start: " << start.time_since_epoch().count() << " processed " << lines << " lines in " << duration.count() / 1000.0 << " milliseconds" << endl;
}

int main() {
    test_tick_processing();
}
