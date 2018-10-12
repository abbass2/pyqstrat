#cell 0
# User code
import pathlib
import pyqstrat as pq

def is_trade(fields): return fields[2] == 'T'
def is_quote(fields) : return fields[2] in ['F', 'N']
def is_open_interest(fields): return fields[2] == 'O'
def is_other(fields): return fields[2] not in ['T', 'F', 'N', 'O']

                    
def create_quote_parser(base_date, headers):
    timestamp_idx = headers.index('timestamp')
    qty_idx = headers.index('quantity')
    price_idx = headers.index('premium')
    bid_offer_idx = headers.index('side')
    id_indices = pq.get_field_indices(['putcall', 'expiration', 'strike'], headers)
    meta_indices = pq.get_field_indices(['info', 'exchange'], headers)
    return pq.TextQuoteParser(is_quote, base_date, timestamp_idx, bid_offer_idx, price_idx, qty_idx, id_indices, meta_indices,
                                    pq.fast_time_milli_parser, "B", "O", 10000.0)
    pq.TextQuoteParser()
                                 
def create_trade_parser(base_date, headers):
    timestamp_idx = headers.index('timestamp')
    qty_idx = headers.index('quantity')
    price_idx = headers.index('premium')
    id_indices = pq.get_field_indices(['putcall', 'expiration', 'strike'], headers)
    meta_indices = pq.get_field_indices(['info', 'exchange'], headers)
    return pq.TextTradeParser(is_trade, base_date, timestamp_idx, price_idx, qty_idx, id_indices, meta_indices,
                                    pq.fast_time_milli_parser, 10000.0)
                                 
def create_open_interest_parser(base_date, headers):
    timestamp_idx = headers.index('timestamp')
    qty_idx = headers.index('quantity')
    id_indices = pq.get_field_indices(['putcall', 'expiration', 'strike'], headers)
    meta_indices = pq.get_field_indices(['info', 'exchange'], headers)
    return pq.TextOpenInterestParser(is_open_interest, base_date, timestamp_idx, qty_idx, id_indices, meta_indices,
                                    pq.fast_time_milli_parser, 10000.0)
                           
def create_other_parser(base_date, headers):
    timestamp_idx = headers.index('timestamp')
    qty_idx = headers.index('quantity')
    id_indices = pq.get_field_indices(['putcall', 'expiration', 'strike'], headers)
    meta_indices = pq.get_field_indices(['info', 'exchange'], headers)
    return pq.TextOtherParser(is_open_interest, base_date, timestamp_idx, id_indices, meta_indices, pq.fast_time_milli_parser)
    
def create_record_parser(quote_parser, trade_parser, open_interest_parser, other_parser):
    return pq.TextRecordParser(quote_parser, trade_parser, open_interest_parser, other_parser)
        
def create_all_quote_aggregator(writer_creator, output_file_prefix):
    return pq.AllQuoteAggregator(writer_creator, output_file_prefix + ".quotes")                                 

def create_all_trade_aggregator(writer_creator, output_file_prefix):
    return pq.AllTradeAggregator(writer_creator, output_file_prefix + ".trades")
                                 
def create_quote_tob_aggregator(writer_creator, output_file_prefix, frequency = '1m'):
    return pq.QuoteTOBAggregator(writer_creator, output_file_prefix + ".tob", frequency = frequency)
                                 
def create_trade_bar_aggregator(writer_creator, output_file_prefix, frequency = '1m'):
    return pq.TradeBarAggregator(writer_creator, output_file_prefix + ".trades", frequency = frequency)
                                 
def create_other_aggregator(writer_creator, output_file_prefix):
    return pq.AllOtherAggregator(writer_creator, output_file_prefix + ".other")
                                 
def create_open_interest_aggregator(writer_creator, output_file_prefix):
    return pq.AllOpenInterestAggregator(writer_creator, output_file_prefix + ".open_interest")

def process(input_filename):
    #output_file_prefix_mapper = pq.SingleDirectoryFileNameMapper(home_dir + '/Developer/coatbridge/histdata/spx_daily_options/daily_1m_bars')
    output_file_prefix_mapper = pq.SingleDirectoryFileNameMapper(home_dir + '/tmp')

    line_filter = pq.SubStringLineFilter([",T,", ",F,", ",N,", ",O,"]) # N indicates there is no available bid or offer
    #with pq.ostream_redirect(stdout=True, stderr=True):
    pq.process_marketdata_file(input_filename,
                 output_file_prefix_mapper, 
                 create_quote_parser,
                 create_trade_parser,
                 create_open_interest_parser,
                 create_other_parser,
                 create_record_parser,
                 create_quote_tob_aggregator,
                 create_trade_bar_aggregator,
                 create_open_interest_aggregator,
                 create_other_aggregator,
                 line_filter)
    
if __name__ == "__main__":
    home_dir = str(pathlib.Path.home())
    #input_filename_provider = PathFileNameProvider("/Users/sal/Developer/coatbridge/vendor_data/algoseek/spx_dailies/*.gz")
    input_filename_provider = pq.PathFileNameProvider("/Users/sal/Developer/coatbridge/vendor_data/algoseek/spx_dailies/w_2018-03-29.csv.gz")
    pq.process_marketdata(input_filename_provider, process, num_processes = 8)

