#ifndef record_types_hpp
#define record_types_hpp

#include <string>
#include <iostream>
#include <pybind11/pybind11.h>

struct ParseException : public std::runtime_error {
    ParseException(const char* m);
};

struct Record {
    virtual ~Record() {}
};

struct TradeRecord : public Record {
    inline explicit TradeRecord(const std::string& id, int64_t timestamp, float qty, float price, const std::string& metadata) :
    id(id), timestamp(timestamp), qty(qty), price(price), metadata(metadata) {}
    std::string id;
    int64_t timestamp;
    float qty;
    float price;
    std::string metadata;
};

struct QuoteRecord : public Record {
    inline explicit QuoteRecord(const std::string& id, int64_t timestamp, bool bid, float qty, float price, const std::string& metadata) :
    id(id), timestamp(timestamp), bid(bid), qty(qty), price(price), metadata(metadata) {}
    std::string id;
    int64_t timestamp;
    bool bid;
    float qty;
    float price;
    std::string metadata;
};


struct QuotePairRecord : public Record {
    inline explicit QuotePairRecord(const std::string& id, int64_t timestamp, float bid_price, float bid_qty,
                            float ask_price, float ask_qty, const std::string& metadata) :
    id(id),
    timestamp(timestamp),
    bid_price(bid_price),
    bid_qty(bid_qty),
    ask_price(ask_price),
    ask_qty(ask_qty),
    metadata(metadata) {}
    
    std::string id;
    int64_t timestamp;
    float bid_price;
    float bid_qty;
    float ask_price;
    float ask_qty;
    std::string metadata;
};

struct OpenInterestRecord : public Record {
    inline explicit OpenInterestRecord(const std::string& id, int64_t timestamp, float qty, const std::string& metadata) :
    id(id), timestamp(timestamp), qty(qty), metadata(metadata) {}
    std::string id;
    int64_t timestamp;
    float qty;
    std::string metadata;
};

struct OtherRecord : public Record {
    inline explicit OtherRecord(const std::string& id, int64_t timestamp, const std::string& metadata) :
    id(id), timestamp(timestamp), metadata(metadata) {}
    std::string id;
    int64_t timestamp;
    std::string metadata;
};

struct Schema {
    enum Type {
        BOOL = 0,
        INT32 = 1,
        INT64 = 2,
        FLOAT32 = 3,
        FLOAT64 = 4,
        STRING = 5,
        TIMESTAMP_MILLI = 6,
        TIMESTAMP_MICRO = 7
    };
    std::vector<std::pair<std::string, Schema::Type>> types;
};

class Tuple {
public:
    template<typename T> void add(const T& val) {
        void* pval = new T(val);
        _pointers.push_back(pval);
        _types.push_back(get_type(val));
    }
    
    template<typename T> const T& get(int i) const {
        return *(reinterpret_cast<T*>(_pointers[i]));
    }
    
    ~Tuple() {
        int i = 0;
        for (auto type : _types) {
            switch (type) {
                case Schema::BOOL:
                    delete reinterpret_cast<bool*>(_pointers[i]);
                    break;
                case Schema::INT32:
                    delete reinterpret_cast<int32_t*>(_pointers[i]);
                    break;
                case Schema::INT64:
                    delete reinterpret_cast<int64_t*>(_pointers[i]);
                    break;
                case Schema::FLOAT32:
                    delete reinterpret_cast<float*>(_pointers[i]);
                    break;
                case Schema::FLOAT64:
                    delete reinterpret_cast<double*>(_pointers[i]);
                    break;
                case Schema::STRING:
                    delete reinterpret_cast<std::string*>(_pointers[i]);
                    break;
                default:
                    std::cerr << "unknown type: " << type << std::endl;;
            }
            i++;
            
        }
        _pointers.clear();
        _types.clear();
    }
    

    
private:
    Schema::Type get_type(const bool&) { return Schema::BOOL; }
    Schema::Type get_type(const int32_t&) { return Schema::INT32; }
    Schema::Type get_type(const int64_t&) { return Schema::INT64; }
    Schema::Type get_type(const float&) { return Schema::FLOAT32; }
    Schema::Type get_type(const double&) { return Schema::FLOAT64; }
    Schema::Type get_type(const std::string&) { return Schema::STRING; }

    std::vector<int> _types;
    std::vector<void*> _pointers;
};

struct Writer {
    virtual void add_record(int line_number, const Tuple&) = 0;
    virtual void write_batch(const std::string& batch_id = "") = 0;
    virtual void close(bool success = true) = 0;
    virtual ~Writer() {};
};


// Base class for functions to avoid using pybind11 binding for functions which has a ton of overhead
// through the pybind1::cpp_function class
template <typename Signature>
class Function;

template <typename Return, typename... Args>
class Function<Return(Args...)> {
public:
    virtual Return call(Args... xs) = 0;
    virtual ~Function() {};
};

class RecordParser {
public:
    virtual void add_line(const std::string& line) = 0;
    virtual std::shared_ptr<Record> parse() = 0;
    virtual ~RecordParser() {};
};

using WriterCreator = Function<std::shared_ptr<Writer>(const std::string&, const Schema&, bool, int)>;
using CheckFields = Function<bool(const std::vector<std::string>&)>;
using TimestampParser = Function<int64_t(const std::string&)>;
using QuoteParser = Function<std::shared_ptr<Record> (const std::vector<std::string>&)>;
using QuotePairParser = Function<std::shared_ptr<Record> (const std::vector<std::string>&)>;
using TradeParser = Function<std::shared_ptr<Record>(const std::vector<std::string>&)>;
using OpenInterestParser = Function<std::shared_ptr<Record>(const std::vector<std::string>&)>;
using OtherParser = Function<std::shared_ptr<Record>(const std::vector<std::string>&)>;
using RecordFieldParser = Function<std::shared_ptr<Record>(const std::vector<std::string>&)>;
using Aggregator = Function<void(const Record*, int)>;
using MissingDataHandler = Function<void(std::shared_ptr<Record>)>;
using BadLineHandler = Function<std::shared_ptr<Record>(int, const std::string&, const std::exception&)>;
using LineFilter = Function<bool(const std::string&)>;
using CheckFields = Function<bool(const std::vector<std::string>&)>;
class StreamHolder;
using RecordGenerator = Function<std::shared_ptr<StreamHolder>(const std::string&, const std::string&)>;
using FileProcessor = Function<int(const std::string&, const std::string& compression)>;
using RecordFilter = Function<bool (const Record &)>;


#endif // record_types_hpp
