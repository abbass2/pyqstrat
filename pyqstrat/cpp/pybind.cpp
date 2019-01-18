#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/iostream.h>

#include "pq_types.hpp"
#include "arrow_writer.hpp"
#include "aggregators.hpp"
#include "text_file_parsers.hpp"
#include "text_file_processor.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std;

// Template for trampoline classes.  See https://pybind11.readthedocs.io/en/stable/advanced/classes.html

template <typename Signature>
class PyFunction;

template <typename Return, typename... Args>
class PyFunction<Return(Args...)> : public Function<Return(Args...)> {
public:
    using Function<Return(Args...)>::Function;
    Return call(Args... args) override {
        PYBIND11_OVERLOAD_PURE_NAME(
                                    Return,
                                    Function<Return(Args...)>,
                                    "__call__",
                                    call,
                                    std::forward<Args>(args)...
                                    );
    }
};

class PyRecordParser : public RecordParser {
public:
    using RecordParser::RecordParser;
    void add_line(const std::string& line) override {
        PYBIND11_OVERLOAD_PURE(void,
                                RecordParser,
                                add_line,
                                line);
    }
    std::shared_ptr<Record> parse() override {
        PYBIND11_OVERLOAD_PURE(std::shared_ptr<Record>,
                                RecordParser,
                                parse);
    }
};

#define TRAMPOLINE(type) \
py::class_<type, Py##type> function_##type (m, #type); \
function_##type \
.def(py::init<>()) \
.def("__call__", & type ::call )

void init_pybind_options(py::module &); //Initialize the black scholes options module

PYBIND11_MODULE(pyqstrat_cpp, m) {
    init_pybind_options(m);
    
    py::add_ostream_redirect(m, "ostream_redirect");
    m.attr("__name__") = "pyqstrat.pyqstrat_cpp";
    py::options options;
    options.disable_function_signatures();
    
    py::class_<RecordParser, PyRecordParser> record_parser (m, "RecordParser");
    record_parser
    .def(py::init<>())
    .def("add_line", & RecordParser::add_line,
         R"pqdoc(
         Args:
            line (str): The line we need to parse
         )pqdoc")
    .def("parse", &RecordParser::parse,
         R"pqdoc(
         Return: a subclass of :obj:`Record`
         )pqdoc");
    
    py::class_<TextRecordParser>(m, "TextRecordParser", record_parser,
    R"pqdoc(
    A helper class that takes in a text line, separates it into a list of fields based on a delimiter, and then uess the parsers
    passed in to try and parse the line as a quote, trade, open interest record or any other type of record
    )pqdoc")
    .def(py::init<std::vector<RecordFieldParser*>, bool, char >(), py::keep_alive<1, 2>(),
         "parsers"_a,
         "exclusive"_a = false,
         "separator"_a = ',',
     R"pqdoc(
     Args:
         parsers: A vector of functions that each take a list of strings as input and returns a subclass of :obj:`Record` or None
         exclusive (bool, optional): Set this when each line can only contain one type of record, after one first parser returns a non
            None object, we will not call other parsers.  Default false
         separator (str, optional):  A single character string.  This is the delimiter we use to separate fields from the text passed in.
            Default ,
         )pqdoc");
   
#define FUNCTION_PROTO(type) \
py::class_<type>(m, #type) \
.def("__call__", &type::call)
    

    // Create Trampoline classes
    using PyWriterCreator = PyFunction<std::shared_ptr<Writer>(const std::string&, const Schema&, bool, int)>;
    using PyCheckFields = PyFunction<bool(const std::vector<std::string>&)>;
    using PyTimestampParser = PyFunction<int64_t(const std::string&)>;
    using PyRecordFieldParser = PyFunction<std::shared_ptr<Record>(const std::vector<std::string>&)>;
    using PyAggregator = PyFunction<void(const Record*, int)>;
    using PyMissingDataHandler = PyFunction<void(std::shared_ptr<Record>)>;
    using PyBadLineHandler = PyFunction<std::shared_ptr<Record>(int, const std::string&, const std::exception&)>;
    using PyLineFilter = PyFunction<bool(const std::string&)>;
    using PyCheckFields = PyFunction<bool(const std::vector<std::string>&)>;
    using PyRecordGenerator = PyFunction<std::shared_ptr<StreamHolder>(const std::string&, const std::string&)>;
    using PyFileProcessor = PyFunction<int(const std::string&, const std::string& compression)>;
    using PyRecordFilter = PyFunction<bool (const Record &)>;
    
    TRAMPOLINE(TimestampParser);
    TRAMPOLINE(RecordFieldParser);
    TRAMPOLINE(Aggregator);
    TRAMPOLINE(MissingDataHandler);
    TRAMPOLINE(BadLineHandler);
    TRAMPOLINE(LineFilter);
    TRAMPOLINE(CheckFields);
    TRAMPOLINE(RecordGenerator);
    TRAMPOLINE(FileProcessor);
    TRAMPOLINE(RecordFilter);
    TRAMPOLINE(WriterCreator);
    
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
    
    py::class_<ArrowWriterCreator, WriterCreator>(m, "ArrowWriterCreator")
    .def(py::init<>())
    .def("__call__", &ArrowWriterCreator::call);
    
    py::class_<TradeBarAggregator, Aggregator>(m, "TradeBarAggregator",
    R"pqdoc(
        Aggregate trade records to create trade bars, given a frequency.  Calculates open, high, low, close, volume, vwap as well as last_update_time
          which is timestamp of the last trade that we processed before the bar ended.
    )pqdoc")
    
    .def(py::init<WriterCreator*, const std::string&, const std::string&, bool, int, Schema::Type>(),
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
    
    .def("__call__", &TradeBarAggregator::call, "trade"_a, "line_number"_a,
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
    

    py::class_<QuoteTOBAggregator, Aggregator>(m, "QuoteTOBAggregator",
        R"pqdoc(
        Aggregate top of book quotes to top of book records.  If you specify a frequency such as "5m", we will calculate a record
        every 5 minutes which has the top of book at the end of that bar.  If no frequency is specified, we will create a top of book
        every time a quote comes in.  We assume that the quotes are all top of book quotes and are written in pairs so we have a bid quote
        followed by a offer quote with the same timestamp or vice versa
        )pqdoc")
    .def(py::init<WriterCreator*, const std::string&, const std::string&, bool, int, Schema::Type>(),
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
         
    .def("__call__", &QuoteTOBAggregator::call, "quote"_a, "line_number"_a,
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

    
    py::class_<AllQuoteAggregator, Aggregator>(m, "AllQuoteAggregator",
    R"pqdoc(
    Writes out every quote we see
    )pqdoc")
    .def(py::init<WriterCreator*, const std::string&, int, Schema::Type>(),
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

    .def("__call__", &AllQuoteAggregator::call,  "quote"_a, "line_number"_a,
         R"pqdoc(
         Add a quote record to be written to disk at some point
         
         Args:
             quote (:obj:`QuoteRecord`):
             line_number (int): The line number of the source file that this trade came from.  Used for debugging
        )pqdoc");
    
    py::class_<AllQuotePairAggregator, Aggregator>(m, "AllQuotePairAggregator",
                                               R"pqdoc(
                                               Writes out every quote pair we find
                                               )pqdoc")
    .def(py::init<WriterCreator*, const std::string&, int, Schema::Type>(),
         "writer_creator"_a,
         "output_file_prefix"_a,
         "batch_size"_a = 10000,
         "timestamp_unit"_a = Schema::TIMESTAMP_MILLI,
         R"pqdoc(
             Args:
                 writer_creator: A function that takes an output_file_prefix, schema, create_batch_id and max_batch_size and returns an object
                    implementing the :obj:`Writer` interface
                 output_file_prefix (str): Path of the output file to create.  The writer and aggregator may add suffixes to this to indicate the kind of data and format the file creates.  E.g. "/tmp/output_file_1"
                 batch_size (int, optional): If set, we will write a batch to disk every time we have this many records queued up.  Defaults to 2.1 billion
                 timestamp_unit (Schema.Type, optional): Whether timestamps are measured as milliseconds or microseconds since the unix epoch.
                 Defaults to Schema.TIMESTAMP_MILLI
         )pqdoc")
    
    .def("__call__", &AllQuotePairAggregator::call,  "quote_pair"_a, "line_number"_a,
         R"pqdoc(
         Add a quote pair record to be written to disk at some point
         
         Args:
            quote_pair (:obj:`QuoteRecord`):
                line_number (int): The line number of the source file that this trade came from.  Used for debugging
         )pqdoc");

   
    py::class_<AllTradeAggregator, Aggregator>(m, "AllTradeAggregator",
    R"pqdoc(
    Writes out every trade we see
    )pqdoc")
    .def(py::init<WriterCreator*, const std::string&, int, Schema::Type>(),
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

    .def("__call__", &AllTradeAggregator::call,  "trade"_a, "line_number"_a,
         R"pqdoc(
         Add a trade record to be written to disk at some point
         
         Args:
             trade (:obj:`TradeRecord`):
             line_number (int): The line number of the source file that this trade came from.  Used for debugging
         )pqdoc");
    
    py::class_<AllOpenInterestAggregator, Aggregator>(m, "AllOpenInterestAggregator",
        R"pqdoc(
        Writes out all open interest records
        )pqdoc")
        .def(py::init<WriterCreator*, const std::string&, int, Schema::Type>(),
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
    
    .def("__call__", &AllOpenInterestAggregator::call,  "oi"_a, "line_number"_a,
        R"pqdoc(
        Add an open interest record to be written to disk at some point
         
        Args:
             oi (:obj:`OpenInterestRecord`):
             line_number (int): The line number of the source file that this trade came from.  Used for debugging
        )pqdoc");
    
    py::class_<AllOtherAggregator, Aggregator>(m, "AllOtherAggregator",
    R"pqdoc(
    Writes out any records that are not trades, quotes or open interest
    )pqdoc")
    .def(py::init<WriterCreator*, const std::string&, int, Schema::Type>(),
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
    
    .def("__call__", &AllOtherAggregator::call,  "other"_a, "line_number"_a,
        R"pqdoc(
        Add a record to be written to disk at some point
         
        Args:
            other (:obj:`OtherRecord`):
            line_number (int): The line number of the source file that this trade came from.  Used for debugging
        )pqdoc");
    
    py::class_<FormatTimestampParser, TimestampParser>(m, "FormatTimestampParser",
        R"pqdoc(
            Helper class that parses timestamps according to the strftime format string passed in.  strftime is slow so
                use :obj:`FixedWithTimeParser` if your timestamp has a fixed format such as "HH:MM:SS...."
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
    
    .def("__call__", &FormatTimestampParser::call, "time"_a,
        R"pqdoc(
             Args:
                time (str): The timestamp to parse
         
             Returns:
                int: Number of millis or micros since epoch
        )pqdoc");
    
    py::class_<TextQuoteParser, RecordFieldParser>(m, "TextQuoteParser",
        R"pqdoc(
          Helper class that parses a quote from a list of fields (strings)
        )pqdoc")
    
    .def(py::init<
         CheckFields*,
         int64_t,
         const std::vector<int>&,
         int,
         int,
         int,
         const std::vector<int>&,
         const std::vector<int>&,
         const std::vector<TimestampParser*>,
         const std::string&,
         const std::string&,
         float,
         bool,
         bool>(),
         py::keep_alive<1, 2>(), //Keep pointer to CheckFields alive while this is alive
         py::keep_alive<1, 9>(), // Keep pointer to timestamp parser alive while this is alive
         "is_quote"_a,
         "base_date"_a,
         "timestamp_indices"_a,
         "bid_offer_idx"_a,
         "price_idx"_a,
         "qty_idx"_a,
         "id_field_indices"_a,
         "meta_field_indices"_a,
         "timestamp_parsers"_a,
         "bid_str"_a,
         "offer_str"_a,
         "price_multiplier"_a = 1.0,
         "strip_id"_a = true,
         "strip_meta"_a = true,
         R"pqdoc(
             Args:
                is_quote: a function that takes a list of strings as input and returns a bool if the fields represent a quote
                base_date (int): if the timestamp in the files does not have a date component, pass in the date as number of millis or micros since the epoch
                timestamp_indices (list of int): index of the timestamp field within the record
                bid_offer_idx (int): index of the field that contains whether this is a bid or offer quote
                price_idx (int): index of the price field
                qty_idx (int): index of the quote size field
                id_field_indices (list of str): indices of the fields identifying an instrument.  For example, for a future this could be symbol and expiry.
                    These fields will be concatenated with a separator and placed in the id field in the record
                meta_field_indices (list of str): indices of additional fields you want to store.  For example, the exchange.
                timestamp_parsers: a vector of functions that take a timestamp as a string and returns number of millis or micros since the epoch
                bid_str (str): if the field indicated in bid_offer_idx matches this string, we consider this quote to be a bid
                offer_str (str): if the field indicated in bid_offer_idx matches this string, we consider this quote to be an offer
                price_multiplier: (float, optional): sometimes the price in a file could be in hundredths of cents, and we divide by this to get dollars.
                    Defaults to 1.0
                strip_id (bool, optional): if we want to strip any whitespace from the id fields before concatenating them.  Defaults to True
                strip_meta (bool, optional):  if we want to strip any whitespace from the meta fields before concatenating them.  Defaults to True
             )pqdoc")
    
        .def("__call__", &TextQuoteParser::call, "fields"_a,
        R"pqdoc(
             Args:
                fields (list of str): A list of fields representing the record
             
             Returns:
                QuoteRecord: Or None if this field is not a quote
             )pqdoc");
    
    py::class_<TextQuotePairParser, RecordFieldParser>(m, "TextQuotePairParser",
                                                       R"pqdoc(
                                                       Helper class that parses a quote containing bid / ask in the same record from a list of fields (strings)
                                                       )pqdoc")
    
    .def(py::init<
         CheckFields*,
         int64_t,
         const std::vector<int>&,
         int,
         int,
         int,
         int,
         const std::vector<int>&,
         const std::vector<int>&,
         const std::vector<TimestampParser*>&,
         float,
         bool,
         bool>(),
         py::keep_alive<1, 2>(), //Keep pointer to CheckFields alive while this is alive
         py::keep_alive<1, 11>(), // Keep pointer to timestamp parser alive while TextQuotePairParser is alive
         "is_quote_pair"_a,
         "base_date"_a,
         "timestamp_indices"_a,
         "bid_price_idx"_a,
         "bid_qty_idx"_a,
         "ask_price_idx"_a,
         "ask_qty_idx"_a,
         "id_field_indices"_a,
         "meta_field_indices"_a,
         "timestamp_parsers"_a,
         "price_multiplier"_a = 1.0,
         "strip_id"_a = true,
         "strip_meta"_a = true,
         R"pqdoc(
         Args:
             is_quote_pair: a function that takes a list of strings as input and returns a bool if the fields represent a quote pair
             base_date (int): if the timestamp in the files does not have a date component, pass in the date as number of millis or micros since the epoch
             timestamp_indices (list of int): Index of the timestamp fields within the record.  For example, date and time could be in different fields.
                We add the result of each timestamp field to get the final timestamp
             bid_price_idx (int): index of the field that contains the bid price
             bid_qty_idx (int): index of the field that contains the bid quantity
             ask_price_idx (int): index of the field that contains the ask price
             ask_qty_idx (int): index of the field that contains the ask quantity
             id_field_indices (list of str): indices of the fields identifying an instrument.  For example, for a future this could be symbol and expiry. These fields will be concatenated with a separator and placed in the id field in the record.
             meta_field_indices (list of str): indices of additional fields you want to store.  For example, the exchange.
             timestamp_parsers: a list of functions that takes a timestamp as a string and returns number of millis or micros since the epoch
             price_multiplier: (float, optional): sometimes the price in a file could be in hundredths of cents, and we divide by this to get dollars. Defaults to 1.0
             strip_id (bool, optional): if we want to strip any whitespace from the id fields before concatenating them.  Defaults to True
             strip_meta (bool, optional):  if we want to strip any whitespace from the meta fields before concatenating them.  Defaults to True
         )pqdoc")
    
    .def("__call__", &TextQuotePairParser::call, "fields"_a,
         R"pqdoc(
         Args:
             fields (list of str): A list of fields representing the record
             Returns:
             QuotePairRecord: Or None if this field is not a quote pair
         )pqdoc");
    
    py::class_<TextTradeParser, RecordFieldParser>(m, "TextTradeParser",
        R"pqdoc(
        Helper class that parses a trade from a list of fields (strings)
        )pqdoc")
    
    .def(py::init<
         CheckFields*,
         int64_t,
         const std::vector<int>&,
         int,
         int,
         const std::vector<int>&,
         const std::vector<int>&,
         const std::vector<TimestampParser*>,
         float,
         bool,
         bool>(),
         py::keep_alive<1, 2>(), //Keep pointer to CheckFields alive while this is alive
         py::keep_alive<1, 9>(), // Keep pointer to timestamp parser alive while this is alive
         "is_trade"_a,
         "base_date"_a,
         "timestamp_indices"_a,
         "price_idx"_a,
         "qty_idx"_a,
         "id_field_indices"_a,
         "meta_field_indices"_a,
         "timestamp_parsers"_a,
         "price_multiplier"_a = 1.0,
         "strip_id"_a = true,
         "strip_meta"_a = true,
     R"pqdoc(
         Args:
             is_trade: A function that takes a list of strings as input and returns a bool if the fields represent a trade
             base_date (int): If the timestamp in the files does not have a date component, pass in the date as number of millis or micros since the epoch
             timestamp_indices (list of int): Index of the timestamp fields within the record.  For example, date and time could be in different fields.
                We add the result of each timestamp field to get the final timestamp
             price_idx (int): Index of the price field
             qty_idx (int): Index of the quote size field
             id_field_indices (list of str): Indices of the fields identifying an instrument.  For example, for a future this could be symbol and expiry.
                These fields will be concatenated with a separator and placed in the id field in the record
             meta_field_indices (list of str): Indices of additional fields you want to store.  For example, the exchange.
             timestamp_parsers: A list of functions that takes a timestamp as a string and returns number of millis or micros since the epoch
             price_multiplier: (float, optional): Sometimes the price in a file could be in hundredths of cents, and we divide by this to get dollars.
                Defaults to 1.0
             strip_id (bool, optional): If we want to strip any whitespace from the id fields before concatenating them.  Defaults to True
             strip_meta (bool, optional):  If we want to strip any whitespace from the meta fields before concatenating them.  Defaults to True
         )pqdoc")
    
    .def("__call__", &TextTradeParser::call, "fields"_a,
    R"pqdoc(
         Args:
            fields (list of str): A list of fields representing the record
         
         Returns:
            TradeRecord: or None if this record is not a trade
    )pqdoc");
    
    py::class_<TextOpenInterestParser, RecordFieldParser>(m, "TextOpenInterestParser",
        R"pqdoc(
        Helper class that parses an open interest record from a list of fields (strings)
        )pqdoc")
    
    .def(py::init<CheckFields*, int64_t,
         const std::vector<int>&,
         int,
         const std::vector<int>&,
         const std::vector<int>&,
         const std::vector<TimestampParser*>,
         bool,
         bool>(),
         py::keep_alive<1, 2>(), //Keep pointer to CheckFields alive while this is alive
         py::keep_alive<1, 8>(), // Keep pointer to timestamp parser alive while this is alive
         "is_open_interest"_a,
         "base_date"_a,
         "timestamp_indices"_a,
         "qty_idx"_a,
         "id_field_indices"_a,
         "meta_field_indices"_a,
         "timestamp_parsers"_a,
         "strip_id"_a = true,
         "strip_meta"_a = true,
         
    R"pqdoc(
         Args:
             is_open_interest: A function that takes a list of strings as input and returns a bool if the fields represent an open interest record
             base_date (int): If the timestamp in the files does not have a date component, pass in the date as number of millis or micros since the epoch
             timestamp_indices (list of int): Index of the timestamp fields within the record.  For example, date and time could be in different fields.
                We add the result of each timestamp field to get the final timestamp
             qty_idx (int): Index of the quote size field
             id_field_indices (list of str): Indices of the fields identifying an instrument.  For example, for a future this could be symbol and expiry.
                These fields will be concatenated with a separator and placed in the id field in the record
             meta_field_indices (list of str): Indices of additional fields you want to store.  For example, the exchange.
             timestamp_parsers: A list of function that takes a timestamp as a string and returns number of millis or micros since the epoch
             strip_id (bool, optional): If we want to strip any whitespace from the id fields before concatenating them.  Defaults to True
             strip_meta (bool, optional):  If we want to strip any whitespace from the meta fields before concatenating them.  Defaults to True
         )pqdoc")
    .def("__call__", &TextOpenInterestParser::call, "fields"_a,
         R"pqdoc(
         Args:
            fields (list of str): A list of fields representing the record
         
         Returns:
            OpenInterestRecord: or None if this record is not an open interest record
         )pqdoc");
    
    py::class_<TextOtherParser, RecordFieldParser>(m, "TextOtherParser",
    R"pqdoc(
        Helper class that parses a record that contains information other than a quote, trade or open interest record
        )pqdoc")
    .def(py::init<
         CheckFields*,
         int64_t,
         const std::vector<int>&,
         const std::vector<int>&,
         const std::vector<int>&,
         const std::vector<TimestampParser*>,
         bool,
         bool>(),
         py::keep_alive<1, 2>(), //Keep pointer to CheckFields alive while this is alive
         py::keep_alive<1, 7>(), // Keep pointer to timestamp parser alive while this is alive
         "is_other"_a,
         "base_date"_a,
         "timestamp_indices"_a,
         "id_field_indices"_a,
         "meta_field_indices"_a,
         "timestamp_parsers"_a,
         "strip_id"_a = true,
         "strip_meta"_a = true,
         R"pqdoc(
             Args:
                 is_other: A function that takes a list of strings as input and returns a bool if we want to parse this record
                 base_date (int): If the timestamp in the files does not have a date component, pass in the date as number of millis or micros since the epoch
                 timestamp_indices (list of int): Index of the timestamp fields within the record.  For example, date and time could be in different fields.
                    We add the result of each timestamp field to get the final timestamp
                 id_field_indices (list of str): Indices of the fields identifying an instrument.  For example, for a future this could be symbol and expiry.
                    These fields will be concatenated with a separator and placed in the id field in the record
                 meta_field_indices (list of str): Indices of additional fields you want to store.  For example, the exchange.
                 timestamp_parsers: A list of functions that take a timestamp as a string and returns number of millis or micros since the epoch
                 strip_id (bool, optional): If we want to strip any whitespace from the id fields before concatenating them.  Defaults to True
                 strip_meta (bool, optional):  If we want to strip any whitespace from the meta fields before concatenating them.  Defaults to True
           )pqdoc")
    
    .def("__call__", &TextOtherParser::call, "fields"_a,
         R"pqdoc(
         Args:
            fields (list of str): a list of fields representing the record
         
         Returns:
            OtherRecord:
         )pqdoc");
    
    py::class_<PrintBadLineHandler, BadLineHandler>(m, "PrintBadLineHandler",
        R"pqdoc(
        A helper class that takes in lines we cannot parse and either prints them and continues or raises an Exception
        )pqdoc")
    .def(py::init<bool>(), "raise"_a = false,
         R"pqdoc(
         Args:
            raise (bool, optional): Whether to raise an exception every time this is called or just print debugging info.  Defaults to False
         )pqdoc")
    
    .def("__call__", &PrintBadLineHandler::call, "line_number"_a, "line"_a, "exception"_a,
         R"pqdoc(
         Args:
             line_number (int): Line number of the input file that corresponds to this line (for debugging)
             line (str): The actual line that failed to parse
             exception (Exception): The exception that caused us to fail to parse this line.
         )pqdoc");
  
    py::class_<RegExLineFilter, LineFilter>(m, "RegExLineFilter",
        R"pqdoc(
        A helper class that filters lines from the input file based on a regular expression.  Note that regular expressions are slow, so
        if you just need to match specific strings, use a string matching filter instead.
        )pqdoc")
    
                                
    .def(py::init<const std::string&>(), "pattern"_a,
         R"pqdoc(
         Args:
            pattern (str): The regex pattern to match.  This follows C++ std::regex pattern matching rules as opposed to python
        )pqdoc")
    
    .def("__call__", &RegExLineFilter::call, "line"_a,
         R"pqdoc(
         Args:
            line (str): The string that the regular expression should match.
         
         Returns:
            bool: Whether the regex matched
         )pqdoc");
    
    py::class_<SubStringLineFilter, LineFilter>(m, "SubStringLineFilter",
        R"pqdoc(
        A helper class that will check if a line matches any of a set of strings
        )pqdoc")
    
    .def(py::init<const std::vector<std::string>&>(), "patterns"_a,
         R"pqdoc(
         Args:
            patterns (list of str): The list of strings to match against
         )pqdoc")
    
    .def("__call__", &SubStringLineFilter::call, "line"_a,
         R"pqdoc(
         Args:
            line (str): We check if any of the patterns are present in this string
         
         Returns:
            bool: Whether any of the patterns were present
         )pqdoc");
    
    py::class_<PriceQtyMissingDataHandler, MissingDataHandler>(m, "PriceQtyMissingDataHandler",
          R"pqdoc(
          A helper class that takes a Record as an input, checks whether its a trade or a quote or any open interest record, and if any of
          the prices or quantities are 0, sets them to NAN
        )pqdoc")
        .def(py::init<>())
        .def("__call__", &PriceQtyMissingDataHandler::call,  "record"_a,
          R"pqdoc(
          Args:
            record:  Any subclass of :obj:`Record`
          )pqdoc");
    
    py::class_<FixedWidthTimeParser, TimestampParser>(m, "FixedWidthTimeParser",
         R"pqdoc(
          A helper class that takes a string formatted as HH:MM:SS.xxx and parses it into number of milliseconds or micros since the beginning of the day
          )pqdoc")
    
    .def(py::init<bool,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int>(),
         "micros"_a = false,
         "years_start"_a = -1,
         "years_size"_a = -1,
         "months_start"_a = -1,
         "months_size"_a = -1,
         "days_start"_a = -1,
         "days_size"_a = -1,
         "hours_start"_a = -1,
         "hours_size"_a = -1,
         "minutes_start"_a = -1,
         "minutes_size"_a = -1,
         "seconds_start"_a = -1,
         "seconds_size"_a = -1,
         "millis_start"_a = -1,
         "millis_size"_a = -1,
         "micros_start"_a = -1,
         "micros_size"_a = -1,
         R"pqdoc(
         Args:
             micros (bool, optional): Whether to return timestamp in millisecs or microsecs since 1970.  Default false
             hours_start (int, optional): index where the hour starts in the timestamp string.  Default -1
             hours_size(int, optional): number of characters used for the hour
             minutes_start (int, optional):
             minutes_size(int, optional):
             seconds_start (int, optional):
             seconds_size(int, optional):
             millis_start (int, optional):
             millis_size(int, optional):
             micros_start (int, optional):
             micros_size(int, optional):
         )pqdoc")
    
          .def("__call__", &FixedWidthTimeParser::call, "time"_a,
          R"pqdoc(
          Args:
            time (str):  A string like "2018-01-01 08:35:22.132"
          
          Return:
            int: Milliseconds or microseconds since Unix epoch
          )pqdoc");

    py::class_<IsFieldInList, CheckFields>(m, "IsFieldInList",
      R"pqdoc(
      Simple utility class to check whether the value of fields[flag_idx] is in any of flag_values
      )pqdoc")
      .def(py::init<int, vector<string>>(), "flag_idx"_a, "flag_values"_a,
      R"pqdoc(
          Args:
             fields: a vector of strings
             flag_idx: the index of fields to check
          )pqdoc")
          .def("__call__", &IsFieldInList::call, "_fields"_a,
          R"pqdoc(
          Args:
            flag_values: a vector of strings containing possible values for the field
          
          Returns:
            a boolean
          )pqdoc");
    
    py::class_<TextFileDecompressor, RecordGenerator>(m, "TextFileDecompressor",
         R"pqdoc(
         A helper function that takes a filename and its compression type, and returns a function that we can use to iterate over lines in that file
         )pqdoc")
         .def(py::init<>())
         .def("__call__", &TextFileDecompressor::call,
          "filename"_a, "compression"_a,
          R"pqdoc(
          Args:
              filename (str):  The file to read
              compression (str): One of "" for uncompressed files, "gzip", "bz2" or "lzip"
          
          Returns:
              A function that takes an empty string as input, and fills in that string.  The function should return False EOF has been reached, True otherwise
          )pqdoc");
          
    py::class_<TextFileProcessor, FileProcessor>(m, "TextFileProcessor",
      R"pqdoc(
      A helper class that takes text based market data files and creates parsed and aggregated quote, trade, open interest, and other files from them.
      )pqdoc")
    
     .def(py::init<
         RecordGenerator*,
         LineFilter*,
         RecordParser*,
         BadLineHandler*,
         RecordFilter*,
         MissingDataHandler*,
         std::vector<Aggregator*>,
         int>(),
        "record_generator"_a,
        "line_filter"_a,
        "record_parser"_a,
        "bad_line_handler"_a,
        "record_filter"_a,
        "missing_data_handler"_a,
        "aggregators"_a,
        "skip_rows"_a = 1,
         R"pqdoc(
         Args:
            record_generator: A function that takes a filename and its compression type, and returns a function that we
                can use to iterate over lines in that file
            line_filter: A function that takes a line (str) as input and returns whether we should parse it or discard it
            record_parser: A function that takes a line (str) as input and returns a :obj:`Record` object
            bad_line_handler: A function that takes a line that failed to parse and returns a :obj:`Record` object or None
            record_filter: A function that takes a parsed Record object and returns whether we should keep it or discard it
            missing_data_handler: A function that takes a parsed Record object and deals with missing data, for example, by converting 0's to NANs
            aggregators: A vector of functions that each take a parsed Record object and aggregate it.
            skip_rows (int, optional): Number of rows to skip in the file before starting to read it.  Defaults to 1 to ignore a header line
            )pqdoc")
         
    .def("__call__", &TextFileProcessor::call, "input_filename"_a, "compression"_a,
         py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
         R"pqdoc(
         Args:
            input_filename (str):  The file to read
            compression (str): One of "" for uncompressed files, "gzip", "bz2" or "lzip"
         
         Returns:
            int: Number of lines processed
         )pqdoc");
}

