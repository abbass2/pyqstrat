#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/iostream.h>

#include "types.hpp"
#include "arrow_writer.hpp"
#include "aggregators.hpp"
#include "text_file_parsers.hpp"
#include "text_file_processor.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(pyqstrat_cpp, m) {
    m.attr("__name__") = "pyqstrat.pyqstrat_cpp";
    py::options options;
    options.disable_function_signatures();
    
    //py::add_ostream_redirect(m, "ostream_redirect");
    
    py::class_<Schema> schema(m, "Schema",
        R"pqdoc(
        Describes a list of field names and data types for writing records to disk
                              
        Attributes:
            types: A list of (str, type) tuples describing a record with the name of each field and its datatype
        )pqdoc");
    
    schema.def(py::init<>())
    .def_readwrite("types", &Schema::types);
    
    py::enum_<Schema::Type>(schema, "Type")
    .value("BOOL", Schema::Type::BOOL)
    .value("INT32", Schema::Type::INT32)
    .value("INT64", Schema::Type::INT64)
    .value("FLOAT32", Schema::Type::FLOAT32)
    .value("FLOAT64", Schema::Type::FLOAT64)
    .value("STRING", Schema::Type::STRING)
    .value("TIMESTAMP_MILLI", Schema::Type::TIMESTAMP_MILLI)
    .value("TIMESTAMP_MICRO", Schema::Type::TIMESTAMP_MICRO)
    .export_values();
    
    py::class_<Record, std::shared_ptr<Record>> record(m, "Record");
    
    py::class_<TradeRecord, std::shared_ptr<TradeRecord>, Record>(m, "TradeRecord",
        R"pqdoc(
        A parsed trade record that we can save to disk
                                                                  
        Attributes:
            id (str): A unique string representing a symbol or instrument id
            timestamp (int): Trade time, in milliseconds or microseconds since 1/1/1970
            qty (float): Trade quantity
            price (float): Trade price
            metadata (str): a string representing any extra information you want to save, such as exchange, or special trade conditions
        )pqdoc")
    .def(py::init<const std::string&, int64_t, float, float, const std::string&>())
    .def_readwrite("id", &TradeRecord::id)
    .def_readwrite("timestamp", &TradeRecord::timestamp)
    .def_readwrite("qty", &TradeRecord::qty)
    .def_readwrite("price", &TradeRecord::price)
    .def_readwrite("metadata", &TradeRecord::metadata);
    
    py::class_<QuoteRecord, std::shared_ptr<QuoteRecord>, Record>(m, "QuoteRecord",
        R"pqdoc(
        A parsed quote record that we can save to disk
                                                                  
        Attributes:
            id (str): Represents a symbol or instrument id, for example, for an option you may concantenate underlying symbol, expiration, strike,
                put or call to uniquely identify the instrument
            timestamp (int): Trade time, in milliseconds or microseconds since 1/1/1970
            bid (bool): If True, this is a bid quote, otherwise it is an offer
            qty (float): Trade quantity
            price (float): Trade price
            metadata (str): A string representing any extra information you want to save, such as exchange, or special trade conditions
        )pqdoc")
    .def(py::init<const std::string&, int64_t, bool, float, float, const std::string&>())
    .def_readwrite("id", &QuoteRecord::id)
    .def_readwrite("timestamp", &QuoteRecord::timestamp)
    .def_readwrite("bid", &QuoteRecord::bid)
    .def_readwrite("qty", &QuoteRecord::qty)
    .def_readwrite("price", &QuoteRecord::price)
    .def_readwrite("metadata", &QuoteRecord::metadata);
    
    py::class_<OpenInterestRecord, std::shared_ptr<OpenInterestRecord>, Record>(m, "OpenInterestRecord",
        R"pqdoc(
        Open interest for a future or option.  Usually one record per instrument at the beginning of the day
                                                                                
        Attributes:
            id (str): Represents a symbol or instrument id, for example, for an option you may concantenate underlying symbol, expiration, strike,
                put or call to uniquely identify the instrument
            timestamp (int): Trade time, in milliseconds or microseconds since 1/1/1970
            qty (float): Trade quantity
            metadata (str): A string representing any extra information you want to save, such as exchange, or special trade conditions
        )pqdoc")
    .def(py::init<const std::string&, int64_t, float, const std::string&>())
    .def_readwrite("id", &OpenInterestRecord::id)
    .def_readwrite("timestamp", &OpenInterestRecord::timestamp)
    .def_readwrite("qty", &OpenInterestRecord::qty)
    .def_readwrite("metadata", &OpenInterestRecord::metadata);
    
    py::class_<OtherRecord, std::shared_ptr<OtherRecord>, Record>(m, "OtherRecord",
        R"pqdoc(
        Any other data you want to store from market data besides trades, quotes and open interest.  You can capture any
        important fields in the metadata attribute
                                                                  
        Attributes:
            id (str): Represents a symbol or instrument id, for example, for an option you may concantenate underlying symbol, expiration, strike,
                put or call to uniquely identify the instrument
            timestamp (int): trade time, in milliseconds or microseconds since 1/1/1970
            metadata (str): a string representing any extra information you want to save, such as exchange, or special trade conditions
        )pqdoc")
    .def(py::init<const std::string&, int64_t, const std::string&>())
    .def_readwrite("id", &OtherRecord::id)
    .def_readwrite("timestamp", &OtherRecord::timestamp)
    .def_readwrite("metadata", &OtherRecord::metadata);
    
    py::class_<Writer, std::shared_ptr<Writer>>(m, "Writer",
        R"pqdoc(
        An abstract class that you subclass to provide an object that can write to disk
        )pqdoc")
    
    .def("write_batch", &Writer::write_batch, "batch_id"_a = "",
        R"pqdoc(
        Write a batch of records to disk.  The batch can have an optional string id so we can
        later retrieve just this batch of records without reading the whole file
         
        Args:
            batch_id (str, optional): An identifier which can later be used to retrieve this batch from disk. Defaults to ""
        )pqdoc")
    
    .def("close", &Writer::close, "success"_a = true,
         R"pqdoc(
         Close the writer and flush any remaining data to disk
         
         Args:
            success (bool, optional): If set to False, we had some kind of exception and are cleaning up.  Tells the function to
                not indicate the file was written successfully, for example by renaming a temp file to the actual filename.  Defaults to True
         )pqdoc");

    py::class_<ArrowWriter, Writer, std::shared_ptr<ArrowWriter>>(m, "ArrowWriter",
    R"pqdoc(
        A subclass of :obj:`Writer` that batches of records to a disk file in the Apache arrow format.  See Apache arrow for details
    )pqdoc")
    
    .def(py::init<const std::string&, const Schema&, bool, int>(),
         "output_file_prefix"_a,
         "schema"_a,
         "create_batch_id_file"_a = false,
         "max_batch_size"_a = -1,
         R"pqdoc(
         Args:
            output_file_prefix (str): Path of the output file to create.  The writer and aggregator may add suffixes to this to indicate the kind
                of data and format the file creates.  E.g. "/tmp/output_file_1"
            schema (:obj:`Schema`): A schema object containing the names and datatypes of each field we want to save in a record
            create_batch_id_file (bool, optional): Whether to create a corresponding file that contains a map from batch id -> batch number
                so we can easily lookup a batch number and then retrieve it from disk.  Defaults to False
            max_batch_size (int, optional): If set, when we get this many records, we write out a batch of records to disk.  May be necessary
                when we are creating large output files, to avoid running out of memory when reading and writing.  Defaults to -1
         )pqdoc")
    
    .def("add_record", &ArrowWriter::add_tuple, "line_number"_a, "tuple"_a,
        R"pqdoc(
        Add a record that will be written to disk at some point.
         
        Args:
            line_number (int): The line number of the source file that this trade came from.  Used for debugging
            tuple (tuple): Must correspond to the schema defined in the constructor.  For example, if the schema has a bool and a float,
                the tuple could be (False, 0.5)
        )pqdoc")
    
    .def("write_batch", &ArrowWriter::write_batch, "batch_id"_a = "",
        R"pqdoc(
            Write a batch of records to disk.  The batch can have an optional string id so we can
            later retrieve just this batch of records without reading the whole file.
         
            Args:
                batch_id (str, optional): An identifier which can later be used to retrieve this batch from disk. Defaults to ""
        )pqdoc")
    
    .def("close", &ArrowWriter::close, "success"_a = true,
         R"pqdoc(
         Close the writer and flush any remaining data to disk
         
         Args:
            success (bool, optional): If set to False, we had some kind of exception and are cleaning up.  Tells the function to
                not indicate the file was written successfully, for example by renaming a temp file to the actual filename.  Defaults to True
        )pqdoc");
    
    py::class_<TradeBarAggregator>(m, "TradeBarAggregator",
    R"pqdoc(
        Aggregate trade records to create trade bars, given a frequency.  Calculates open, high, low, close, volume, vwap as well as last_update_time
          which is timestamp of the last trade that we processed before the bar ended.
    )pqdoc")
    
    .def(py::init<WriterCreator, const std::string&, const std::string&, bool, int, Schema::Type>(),
         "writer_creator"_a,
         "output_file_prefix"_a,
         "frequency"_a = "5m",
         "batch_by_id"_a = true,
         "batch_size"_a = std::numeric_limits<int>::max(),
         "timestamp_unit"_a = Schema::TIMESTAMP_MILLI,
         R"pqdoc(
             Args:
                writer_creator: A function that takes an output_file_prefix, schema, create_batch_id and max_batch_size and returns an object
                    implementing the :obj:`Writer` interface
                output_file_prefix (str): Path of the output file to create.  The writer and aggregator may add suffixes to this to indicate the kind
                    of data and format the file creates.  E.g. "/tmp/output_file_1"
                frequency (str, optional): A string like "5m" indicating the bar size is 5 mins.  Units can be s,m,h or d for second, minute, hour or day.
                    Defaults to "5m"
                batch_by_id (bool, optional): If set, we will create one batch for each id.  This will allow us to retrieve all records for a single
                    instrument by reading a single batch.  Defaults to True.
                batch_size (int, optional): If set, we will write a batch to disk every time we have this many records queued up.  Defaults to 2.1 billion
                timestamp_unit (Schema.Type, optional): Whether timestamps are measured as milliseconds or microseconds since the unix epoch.
                    Defaults to Schema.TIMESTAMP_MILLI
                )pqdoc")
    
    .def("__call__", &TradeBarAggregator::operator(), "trade"_a, "line_number"_a,
        R"pqdoc(
            Add a trade record to be written to disk at some point
         
            Args:
                trade ( :obj:`TradeRecord`):
                line_number (int): The line number of the source file that this trade came from.  Used for debugging
        )pqdoc")
    
    .def("close", &TradeBarAggregator::close,
        R"pqdoc(
            Flush all unwritten records to the Writer, which writes them to disk when its close function is called
        )pqdoc");
    

    py::class_<QuoteTOBAggregator>(m, "QuoteTOBAggregator",
        R"pqdoc(
        Aggregate top of book quotes to top of book records.  If you specify a frequency such as "5m", we will calculate a record
        every 5 minutes which has the top of book at the end of that bar.  If no frequency is specified, we will create a top of book
        every time a quote comes in.  We assume that the quotes are all top of book quotes and are written in pairs so we have a bid quote
        followed by a offer quote with the same timestamp or vice versa
        )pqdoc")
    .def(py::init<WriterCreator, const std::string&, const std::string&, bool, int, Schema::Type>(),
         "writer_creator"_a,
         "output_file_prefix"_a,
         "frequency"_a = "5m",
         "batch_by_id"_a = true,
         "batch_size"_a = std::numeric_limits<int>::max(),
         "timestamp_unit"_a = Schema::TIMESTAMP_MILLI,
         R"pqdoc(
         Args:
            writer_creator: A function that takes an output_file_prefix, schema, create_batch_id and max_batch_size and returns an object
                implementing the :obj:`Writer` interface
            output_file_prefix (str): Path of the output file to create.  The writer and aggregator may add suffixes to this to indicate the kind
                of data and format the file creates.  E.g. "/tmp/output_file_1"
            frequency (str, optional): A string like "5m" indicating the bar size is 5 mins.  Units can be s,m,h or d for second, minute, hour or day.
                Defaults to "5m".  If you set this to "", each tick will be recorded.
            batch_by_id (bool, optional): If set, we will create one batch for each id.  This will allow us to retrieve all records for a single
            instrument by reading a single batch.  Defaults to True.
            batch_size (int, optional): If set, we will write a batch to disk every time we have this many records queued up.  Defaults to 2.1 billion
            timestamp_unit (Schema.Type, optional): Whether timestamps are measured as milliseconds or microseconds since the unix epoch.
                Defaults to Schema.TIMESTAMP_MILLI
            )pqdoc")
         
    .def("__call__", &QuoteTOBAggregator::operator(), "quote"_a, "line_number"_a,
         R"pqdoc(
         Add a quote record to be written to disk at some point
         
         Args:
             quote ( :obj:`QuoteRecord`):
             line_number (int): The line number of the source file that this trade came from.  Used for debugging
         )pqdoc")
    .def("close", &QuoteTOBAggregator::close,
         R"pqdoc(
         Flush all unwritten records to the Writer, which writes them to disk when its close function is called
         )pqdoc");

    
    py::class_<AllQuoteAggregator>(m, "AllQuoteAggregator",
    R"pqdoc(
    Writes out every quote we see
    )pqdoc")
    .def(py::init<WriterCreator, const std::string&, int, Schema::Type>(),
         "writer_creator"_a,
         "output_file_prefix"_a,
         "batch_size"_a = 10000,
         "timestamp_unit"_a = Schema::TIMESTAMP_MILLI,
         R"pqdoc(
             Args:
                 writer_creator: A function that takes an output_file_prefix, schema, create_batch_id and max_batch_size and returns an object
                    implementing the :obj:`Writer` interface
                 output_file_prefix (str): Path of the output file to create.  The writer and aggregator may add suffixes to this to indicate the kind
                    of data and format the file creates.  E.g. "/tmp/output_file_1"
                 batch_size (int, optional): If set, we will write a batch to disk every time we have this many records queued up.  Defaults to 2.1 billion
                timestamp_unit (Schema.Type, optional): Whether timestamps are measured as milliseconds or microseconds since the unix epoch.
                    Defaults to Schema.TIMESTAMP_MILLI
         )pqdoc")

    .def("__call__", &AllQuoteAggregator::operator(),  "quote"_a, "line_number"_a,
         R"pqdoc(
         Add a quote record to be written to disk at some point
         
         Args:
             quote (:obj:`QuoteRecord`):
             line_number (int): The line number of the source file that this trade came from.  Used for debugging
        )pqdoc");
   
    py::class_<AllTradeAggregator>(m, "AllTradeAggregator",
    R"pqdoc(
    Writes out every trade we see
    )pqdoc")
    .def(py::init<WriterCreator, const std::string&, int, Schema::Type>(),
         "writer_creator"_a,
         "output_file_prefix"_a,
         "batch_size"_a = 10000,
         "timestamp_unit"_a = Schema::TIMESTAMP_MILLI,
    R"pqdoc(
    Args:
        writer_creator: A function that takes an output_file_prefix, schema, create_batch_id and max_batch_size and returns an object
             implementing the :obj:`Writer` interface
        output_file_prefix (str): Path of the output file to create.  The writer and aggregator may add suffixes to this to indicate the kind
             of data and format the file creates.  E.g. "/tmp/output_file_1"
        batch_size (int, optional): If set, we will write a batch to disk every time we have this many records queued up.  Defaults to 2.1 billion
        timestamp_unit (Schema.Type, optional): Whether timestamps are measured as milliseconds or microseconds since the unix epoch.
            Defaults to Schema.TIMESTAMP_MILLI
    )pqdoc")

    .def("__call__", &AllTradeAggregator::operator(),  "trade"_a, "line_number"_a,
         R"pqdoc(
         Add a trade record to be written to disk at some point
         
         Args:
             trade (:obj:`TradeRecord`):
             line_number (int): The line number of the source file that this trade came from.  Used for debugging
         )pqdoc");
    
    py::class_<AllOpenInterestAggregator>(m, "AllOpenInterestAggregator",
        R"pqdoc(
        Writes out all open interest records
        )pqdoc")
        .def(py::init<WriterCreator, const std::string&, int, Schema::Type>(),
             "writer_creator"_a,
             "output_file_prefix"_a,
             "batch_size"_a = 10000,
             "timestamp_unit"_a = Schema::TIMESTAMP_MILLI,
         R"pqdoc(
     Args:
         writer_creator: A function that takes an output_file_prefix, schema, create_batch_id and max_batch_size and returns an object
            implementing the :obj:`Writer` interface
         output_file_prefix (str): Path of the output file to create.  The writer and aggregator may add suffixes to this to indicate the kind
            of data and format the file creates.  E.g. "/tmp/output_file_1"
         batch_size (int, optional): If set, we will write a batch to disk every time we have this many records queued up.  Defaults to 2.1 billion
         timestamp_unit (Schema.Type, optional): Whether timestamps are measured as milliseconds or microseconds since the unix epoch.
            Defaults to Schema.TIMESTAMP_MILLI
     )pqdoc")
    
    .def("__call__", &AllOpenInterestAggregator::operator(),  "oi"_a, "line_number"_a,
        R"pqdoc(
        Add an open interest record to be written to disk at some point
         
        Args:
             oi (:obj:`OpenInterestRecord`):
             line_number (int): The line number of the source file that this trade came from.  Used for debugging
        )pqdoc");
    
    py::class_<AllOtherAggregator>(m, "AllOtherAggregator",
    R"pqdoc(
    Writes out any records that are not trades, quotes or open interest
    )pqdoc")
    .def(py::init<WriterCreator, const std::string&, int, Schema::Type>(),
         "writer_creator"_a,
         "output_file_prefix"_a,
         "batch_size"_a = 10000,
         "timestamp_unit"_a = Schema::TIMESTAMP_MILLI,
         R"pqdoc(
         Args:
             writer_creator: A function that takes an output_file_prefix, schema, create_batch_id and max_batch_size and returns an object
                implementing the :obj:`Writer` interface
             output_file_prefix (str): Path of the output file to create.  The writer and aggregator may add suffixes to this to indicate the kind
                of data and format the file creates.  E.g. "/tmp/output_file_1"
             batch_size (int, optional): If set, we will write a batch to disk every time we have this many records queued up.  Defaults to 2.1 billion
             timestamp_unit (Schema.Type, optional): Whether timestamps are measured as milliseconds or microseconds since the unix epoch.
                Defaults to Schema.TIMESTAMP_MILLI
         )pqdoc")
    
    .def("__call__", &AllOtherAggregator::operator(),  "other"_a, "line_number"_a,
        R"pqdoc(
        Add a record to be written to disk at some point
         
        Args:
            other (:obj:`OtherRecord`):
            line_number (int): The line number of the source file that this trade came from.  Used for debugging
        )pqdoc");
    
    py::class_<FormatTimestampParser>(m, "FormatTimestampParser",
        R"pqdoc(
            Helper class that parses timestamps according to the strftime format string passed in.  strftime is slow so use fast_milli_time_parser
                or fast_time_micro_parser if your timestamps are in "HH:MM:SS...." format
        )pqdoc")

    .def(py::init<int64_t,
         const std::string&,
         bool>(),
        "base_date"_a,
        "time_format"_a = "%H:%M:%S",
        "micros"_a = false,
        R"pqdoc(
             Args:
                base_date (int): Sometimes the timestamps in a file contain time only and the name of a file contains the date.  In these cases, pass in the date
                    as number of millis or micros from the epoch to the start of that date.  If the timestamp has date also, pass in 0 here.
                time_format (str, optional): strftime format string for parsing the timestamp.  Defaults to "%H:%M:%S"
                micros (bool, optional): If this is set, we will parse and store microseconds.  Otherwise we will parse and store milliseconds.
                    Defaults to True
             )pqdoc")
    
    .def("__call__", &FormatTimestampParser::operator(), "time"_a,
        R"pqdoc(
             Args:
                time (str): The timestamp to parse
         
             Returns:
                int: Number of millis or micros since epoch
        )pqdoc");
    
    py::class_<TextQuoteParser, std::shared_ptr<TextQuoteParser>>(m, "TextQuoteParser",
        R"pqdoc(
          Helper class that parses a quote from a list of fields (strings)
        )pqdoc")
    
    .def(py::init<
         IsRecordFunc,
         int64_t,
         int,
         int,
         int,
         int,
         const std::vector<int>&,
         const std::vector<int>&,
         TimestampParser,
         const std::string&,
         const std::string&,
         float,
         bool,
         bool>(),
         "is_quote"_a,
         "base_date"_a,
         "timestamp_idx"_a,
         "bid_offer_idx"_a,
         "price_idx"_a,
         "qty_idx"_a,
         "id_field_indices"_a,
         "meta_field_indices"_a,
         "timestamp_parser"_a,
         "bid_str"_a,
         "offer_str"_a,
         "price_multiplier"_a = 1.0,
         "strip_id"_a = true,
         "strip_meta"_a = true,
         R"pqdoc(
             Args:
                is_quote: a function that takes a list of strings as input and returns a bool if the fields represent a quote
                base_date (int): if the timestamp in the files does not have a date component, pass in the date as number of millis or micros since the epoch
                timestamp_idx (int): index of the timestamp field within the record
                bid_offer_idx (int): index of the field that contains whether this is a bid or offer quote
                price_idx (int): index of the price field
                qty_idx (int): index of the quote size field
                id_field_indices (list of str): indices of the fields identifying an instrument.  For example, for a future this could be symbol and expiry.
                    These fields will be concatenated with a separator and placed in the id field in the record
                meta_field_indices (list of str): indices of additional fields you want to store.  For example, the exchange.
                timestamp_parser: a function that takes a timestamp as a string and returns number of millis or micros since the epoch
                bid_str (str): if the field indicated in bid_offer_idx matches this string, we consider this quote to be a bid
                offer_str (str): if the field indicated in bid_offer_idx matches this string, we consider this quote to be an offer
                price_multiplier: (float, optional): sometimes the price in a file could be in hundredths of cents, and we divide by this to get dollars.
                    Defaults to 1.0
                strip_id (bool, optional): if we want to strip any whitespace from the id fields before concatenating them.  Defaults to True
                strip_meta (bool, optional):  if we want to strip any whitespace from the meta fields before concatenating them.  Defaults to True
             )pqdoc")
    
        .def("__call__", &TextQuoteParser::operator(), "fields"_a,
        R"pqdoc(
             Args:
                fields (list of str): A list of fields representing the record
             
             Returns:
                QuoteRecord: Or None if this field is not a quote
             )pqdoc");
    
    py::class_<TextTradeParser, std::shared_ptr<TextTradeParser>>(m, "TextTradeParser",
        R"pqdoc(
        Helper class that parses a trade from a list of fields (strings)
        )pqdoc")
    
    .def(py::init<IsRecordFunc, int64_t, int, int, int,
         const std::vector<int>&,
         const std::vector<int>&,
         TimestampParser,
         float,
         bool,
         bool>(),
         "is_trade"_a,
         "base_date"_a,
         "timestamp_idx"_a,
         "price_idx"_a,
         "qty_idx"_a,
         "id_field_indices"_a,
         "meta_field_indices"_a,
         "timestamp_parser"_a,
         "price_multiplier"_a = 1.0,
         "strip_id"_a = true,
         "strip_meta"_a = true,
     R"pqdoc(
         Args:
             is_trade: A function that takes a list of strings as input and returns a bool if the fields represent a trade
             base_date (int): If the timestamp in the files does not have a date component, pass in the date as number of millis or micros since the epoch
             timestamp_idx (int): Index of the timestamp field within the record
             price_idx (int): Index of the price field
             qty_idx (int): Index of the quote size field
             id_field_indices (list of str): Indices of the fields identifying an instrument.  For example, for a future this could be symbol and expiry.
                These fields will be concatenated with a separator and placed in the id field in the record
             meta_field_indices (list of str): Indices of additional fields you want to store.  For example, the exchange.
             timestamp_parser: A function that takes a timestamp as a string and returns number of millis or micros since the epoch
             price_multiplier: (float, optional): Sometimes the price in a file could be in hundredths of cents, and we divide by this to get dollars.
                Defaults to 1.0
             strip_id (bool, optional): If we want to strip any whitespace from the id fields before concatenating them.  Defaults to True
             strip_meta (bool, optional):  If we want to strip any whitespace from the meta fields before concatenating them.  Defaults to True
    )pqdoc")
         
    .def("__call__", &TextTradeParser::operator(), "fields"_a,
    R"pqdoc(
         Args:
            fields (list of str): A list of fields representing the record
         
         Returns:
            TradeRecord: or None if this record is not a trade
    )pqdoc");
    
    py::class_<TextOpenInterestParser, std::shared_ptr<TextOpenInterestParser>>(m, "TextOpenInterestParser",
        R"pqdoc(
        Helper class that parses an open interest record from a list of fields (strings)
        )pqdoc")
    
    .def(py::init<IsRecordFunc, int64_t, int, int,
         const std::vector<int>&,
         const std::vector<int>&,
         TimestampParser,
         bool,
         bool>(),
         "is_open_interest"_a,
         "base_date"_a,
         "timestamp_idx"_a,
         "qty_idx"_a,
         "id_field_indices"_a,
         "meta_field_indices"_a,
         "timestamp_parser"_a,
         "strip_id"_a = true,
         "strip_meta"_a = true,
         
    R"pqdoc(
         Args:
             is_open_interest: A function that takes a list of strings as input and returns a bool if the fields represent an open interest record
             base_date (int): If the timestamp in the files does not have a date component, pass in the date as number of millis or micros since the epoch
             timestamp_idx (int): Index of the timestamp field within the record
             qty_idx (int): Index of the quote size field
             id_field_indices (list of str): Indices of the fields identifying an instrument.  For example, for a future this could be symbol and expiry.
                These fields will be concatenated with a separator and placed in the id field in the record
             meta_field_indices (list of str): Indices of additional fields you want to store.  For example, the exchange.
             timestamp_parser: A function that takes a timestamp as a string and returns number of millis or micros since the epoch
             strip_id (bool, optional): If we want to strip any whitespace from the id fields before concatenating them.  Defaults to True
             strip_meta (bool, optional):  If we want to strip any whitespace from the meta fields before concatenating them.  Defaults to True
         )pqdoc")
    .def("__call__", &TextOpenInterestParser::operator(), "fields"_a,
         R"pqdoc(
         Args:
            fields (list of str): A list of fields representing the record
         
         Returns:
            OpenInterestRecord: or None if this record is not an open interest record
         )pqdoc");
    
    py::class_<TextOtherParser, std::shared_ptr<TextOtherParser>>(m, "TextOtherParser",
    R"pqdoc(
        Helper class that parses a record that contains information other than a quote, trade or open interest record
        )pqdoc")
    .def(py::init<IsRecordFunc,
         int64_t,
         int,
         const std::vector<int>&,
         const std::vector<int>&,
         TimestampParser,
         bool,
         bool>(),
         "is_other"_a,
         "base_date"_a,
         "timestamp_idx"_a,
         "id_field_indices"_a,
         "meta_field_indices"_a,
         "timestamp_parser"_a,
         "strip_id"_a = true,
         "strip_meta"_a = true,
         R"pqdoc(
             Args:
                 is_other: A function that takes a list of strings as input and returns a bool if we want to parse this record
                 base_date (int): If the timestamp in the files does not have a date component, pass in the date as number of millis or micros since the epoch
                 timestamp_idx (int): Index of the timestamp field within the record
                 id_field_indices (list of str): Indices of the fields identifying an instrument.  For example, for a future this could be symbol and expiry.
                    These fields will be concatenated with a separator and placed in the id field in the record
                 meta_field_indices (list of str): Indices of additional fields you want to store.  For example, the exchange.
                 timestamp_parser: A function that takes a timestamp as a string and returns number of millis or micros since the epoch
                 strip_id (bool, optional): If we want to strip any whitespace from the id fields before concatenating them.  Defaults to True
                 strip_meta (bool, optional):  If we want to strip any whitespace from the meta fields before concatenating them.  Defaults to True
           )pqdoc")
    
    .def("__call__", &TextOtherParser::operator(), "fields"_a,
         R"pqdoc(
         Args:
            fields (list of str): a list of fields representing the record
         
         Returns:
            OtherRecord:
         )pqdoc");
    
    py::class_<TextRecordParser>(m, "TextRecordParser",
        R"pqdoc(
        A helper class that takes in a text line, separates it into a list of fields based on a delimiter, and then uess the parsers
        passed in to try and parse the line as a quote, trade, open interest record or any other info
        )pqdoc")
    .def(py::init<
         std::shared_ptr<TextQuoteParser>,
         std::shared_ptr<TextTradeParser>,
         std::shared_ptr<TextOpenInterestParser>,
         std::shared_ptr<TextOtherParser>,
         char>(),
         "quote_parser"_a,
         "trade_parser"_a,
         "open_interest_parser"_a,
         "other_parser"_a,
         "separator"_a = ',',
         R"pqdoc(
         Args:
             quote_parser: A function that takes a list of strings as input and returns either a :obj:`QuoteRecord` or None
             trade_parser: A function that takes a list of strings as input and returns either a :obj:`TradeRecord` or None
             open_interest_parser: A function that takes a list of strings as input and returns either an :obj:`OpenInterest` or None
             other_parser: A function that takes a list of strings as input and returns an :obj:`OtherRecord` or None
             separator (str, optional):  A single character string.  This is the delimiter we use to separate fields from the text passed in
         )pqdoc")
    .def("__call__", &TextRecordParser::operator(), "line"_a,
         R"pqdoc(
         Args:
             line (str): The line we need to parse
         )pqdoc");
    
    py::class_<PrintBadLineHandler>(m, "PrintBadLineHandler",
        R"pqdoc(
        A helper class that takes in lines we cannot parse and either prints them and continues or raises an Exception
        )pqdoc")
    .def(py::init<bool>(), "raise"_a = false,
         R"pqdoc(
         Args:
            raise (bool, optional): Whether to raise an exception every time this is called or just print debugging info.  Defaults to False
         )pqdoc")
    
    .def("__call__", &PrintBadLineHandler::operator(), "line_number"_a, "line"_a, "exception"_a,
         R"pqdoc(
         Args:
             line_number (int): Line number of the input file that corresponds to this line (for debugging)
             line (str): The actual line that failed to parse
             exception (Exception): The exception that caused us to fail to parse this line.
         )pqdoc");
  
    py::class_<RegExLineFilter>(m, "RegExLineFilter",
        R"pqdoc(
        A helper class that filters lines from the input file based on a regular expression.  Note that regular expressions are slow, so
        if you just need to match specific strings, use a string matching filter instead.
        )pqdoc")
                                
    .def(py::init<const std::string&>(), "pattern"_a,
         R"pqdoc(
         Args:
            pattern (str): The regex pattern to match.  This follows C++ std::regex pattern matching rules as opposed to python
        )pqdoc")
    
    .def("__call__", &RegExLineFilter::operator(), "line"_a,
         R"pqdoc(
         Args:
            line (str): The string that the regular expression should match.
         
         Returns:
            bool: Whether the regex matched
         )pqdoc");
    
    py::class_<SubStringLineFilter>(m, "SubStringLineFilter",
        R"pqdoc(
        A helper class that will check if a line matches any of a set of strings
        )pqdoc")
    
    .def(py::init<const std::vector<std::string>&>(), "patterns"_a,
         R"pqdoc(
         Args:
            patterns (list of str): The list of strings to match against
         )pqdoc")
    
    .def("__call__", &SubStringLineFilter::operator(), "line"_a,
         R"pqdoc(
         Args:
            line (str): We check if any of the patterns are present in this string
         
         Returns:
            bool: Whether any of the patterns were present
         )pqdoc");
    
    m.def("price_qty_missing_data_handler", price_qty_missing_data_handler, "record"_a,
          R"pqdoc(
          A helper function that takes a Record as an input, checks whether its a trade or a quote or any open interest record, and if any of
          the prices or quantities are 0, sets them to NAN
          
          Args:
            record:  Any subclass of :obj:`Record`
          )pqdoc");
    
    /* m.def("is_quote", is_quote, "fields"_a);
    m.def("is_trade", is_trade, "fields"_a);
    m.def("is_open_interest", is_open_interest, "fields"_a);
    m.def("is_other", is_other, "fields"_a); */
    
    m.def("fast_time_milli_parser", fast_time_milli_parser, "time"_a,
          R"pqdoc(
          A helper function that takes a string formatted as HH:MM:SS.xxx and parses it into number of milliseconds since the beginning of the day
          
          Args:
            time (str):  A string like "08:35:22.132"
          
          Returns:
            int: Millis since beginning of day
          )pqdoc");

    m.def("fast_time_micro_parser", fast_time_micro_parser, "time"_a,
          R"pqdoc(
          A helper function that takes a string formatted as HH:MM:SS.xxxxxx and parses it into number of microseconds since the beginning of the day
          
          Args:
            time (str):  A string like "08:35:22.132876"
          
          Returns:
            int: Microseconds since beginning of day
          )pqdoc");
    
    m.def("is_field_in_list", is_field_in_list, "fields"_a, "flag_idx"_a, "flag_values"_a,
          R"pqdoc(
          Simple utility function to check whether the value of fields[flag_idx] is in any of flag_values
          
          Args:
            fields: a vector of strings
            flag_idx: the index of fields to check
            flag_values: a vector of strings containing possible values for the field
          
          Returns:
            a boolean
          )pqdoc");
    
    m.def("text_file_decompressor", text_file_decompressor, "filename"_a, "compression"_a,
          R"pqdoc(
          A helper function that takes a filename and its compression type, and returns a function that we can use to iterate over lines in that file
          
          Args:
              filename (str):  The file to read
              compression (str): One of "" for uncompressed files, "gzip", "bz2" or "lzip"
          
          Returns:
              A function that takes an empty string as input, and fills in that string.  The function should return False EOF has been reached, True otherwise
          )pqdoc");
          
    py::class_<TextFileProcessor>(m, "TextFileProcessor")
    .def(py::init<
        std::function<std::shared_ptr<StreamHolder>(const std::string&, const std::string&)>,
        std::function<bool (const std::string&)>,
        std::function<std::shared_ptr<Record> (const std::string&)>,
        std::function<std::shared_ptr<Record> (int, const std::string&, const std::exception&)>,
        std::function<bool (const Record&)>,
        std::function<void (std::shared_ptr<Record>)>,
        std::function<void (const QuoteRecord&, int)>,
        std::function<void (const TradeRecord&, int)>,
        std::function<void (const OpenInterestRecord&, int)>,
        std::function<void (const OtherRecord&, int)>,
        int>(),
        "record_generator"_a,
        "line_filter"_a,
        "record_parser"_a,
        "bad_line_handler"_a,
        "record_filter"_a,
        "missing_data_handler"_a,
        "quote_aggregator"_a,
        "trade_aggregator"_a,
        "open_interest_aggregator"_a,
        "other_aggregator"_a,
        "skip_rows"_a = 1,
         R"pqdoc(
         A helper class that takes text based market data files and creates parsed and aggregated quote, trade, open interest, and other files from them.
         
         Args:
            record_generator: A function that takes a filename and its compression type, and returns a function that we
                can use to iterate over lines in that file
            line_filter: A function that takes a line (str) as input and returns whether we should parse it or discard it
            record_parser: A function that takes a line (str) as input and returns a :obj:`Record` object
            bad_line_handler: A function that takes a line that failed to parse and returns a :obj:`Record` object or None
            record_filter: A function that takes a parsed Record object and returns whether we should keep it or discard it
            missing_data_handler: A function that takes a parsed Record object and deals with missing data, for example, by converting 0's to NANs
            quote_aggregator: A function that takes parsed quote objects and aggregates them
            trade_aggregator: A function that takes parsed trade objects and aggregates them
            open_interest_aggregator: A function that takes parsed open interest objects and aggregates them
            other_aggregator: A function that takes parsed :obj:`OtherRecord` objects and aggregates them
            skip_rows (int, optional): Number of rows to skip in the file before starting to read it.  Defaults to 1 to ignore a header line
            )pqdoc")
         
    .def("__call__", &TextFileProcessor::operator(), "input_filename"_a, "compression"_a,
         py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
         R"pqdoc(
         Args:
            input_filename (str):  The file to read
            compression (str): One of "" for uncompressed files, "gzip", "bz2" or "lzip"
         
         Returns:
            int: Number of lines processed
         )pqdoc");
}

