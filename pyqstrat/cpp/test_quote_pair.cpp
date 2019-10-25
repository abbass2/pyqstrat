
#include <chrono>
#include "hdf5_writer.hpp"
#include "text_file_parsers.hpp"
#include "aggregators.hpp"
#include "text_file_processor.hpp"
#include "py_import_call_execute.hpp"
#include "file_reader.hpp"
#include "H5Cpp.h"

using namespace std::chrono;
using namespace std;

void test_quote_pair_processing() {
    cout << "starting" << endl;
    auto timestamp_parser = FixedWidthTimeParser(false, 0, 2, 3, 2);
    auto quote_pair_parser = make_shared<TextQuotePairParser>(nullptr, 0,
                                                              vector<int>{8}, 37, 38, 40, 41, vector<int>{0},
                                                              vector<int>{}, vector<TimestampParser*>{&timestamp_parser}, 1.0);
    RecordFieldParser* parser = quote_pair_parser.get();
    std::vector<RecordFieldParser*> parsers = {parser};
    auto text_record_parser = TextRecordParser(parsers, false);
    auto hdf5_writer_creator = HDF5WriterCreator("/tmp/quotes");
    auto quote_pair_aggregator = AllQuotePairAggregator(&hdf5_writer_creator);
    std::vector<Aggregator*> aggregators = {&quote_pair_aggregator};
    auto text_file_decompressor = TextFileDecompressor();
    auto print_bad_line_handler = PrintBadLineHandler();
    auto price_qty_missing_data_handler = PriceQtyMissingDataHandler();
    auto processor = TextFileProcessor(&text_file_decompressor,
                                       nullptr,
                                       &text_record_parser,
                                       &print_bad_line_handler,
                                       nullptr,
                                       &price_qty_missing_data_handler,
                                       aggregators,
                                       1);
    auto start = high_resolution_clock::now();
    int lines = processor.call("/Users/sal/tmp/date_gz/20181203/20181203_E1CZ8.P2190.csv.gz", "gzip");
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    cout << "start: " << start.time_since_epoch().count() << " processed " << lines << " lines in " << duration.count() / 1000.0 << " milliseconds" << endl;
    
    start = high_resolution_clock::now();
    lines = processor.call("/Users/sal/tmp/date_gz/20160317/20160317_EW2J6.P1465.csv.gz", "gzip");
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    
    cout << "start: " << start.time_since_epoch().count() << " processed " << lines << " lines in " << duration.count() / 1000.0 << " milliseconds" << endl;
}
