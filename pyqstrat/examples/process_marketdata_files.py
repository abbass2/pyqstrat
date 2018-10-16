#cell 2
import pathlib
import pyqstrat as pq
import tempfile
import os

open_interest_records = \
b'''Timestamp,Ticker,Type,Side,Info,PutCall,Expiration,Strike,Quantity,Premium,Exchange
06:30:31.461,SPXW,O, , ,C,20160316,21000000,1029,0,W
06:30:31.461,SPXW,O, , ,P,20160316,20700000,5,0,W
06:30:31.461,SPXW,O, , ,C,20160316,19750000,205,0,W
06:30:31.461,SPXW,O, , ,C,20160316,19600000,320,0,W
06:30:31.461,SPXW,O, , ,C,20160316,19450000,256,0,W
06:30:31.461,SPXW,O, , ,C,20160316,19300000,9,0,W
06:30:31.461,SPXW,O, , ,C,20160316,19150000,3,0,W
06:30:31.461,SPXW,O, , ,C,20160316,19000000,253,0,W
06:30:31.461,SPXW,O, , ,C,20160316,18850000,5,0,W
06:30:31.462,SPXW,O, , ,P,20160316,18650000,194,0,W
06:30:31.463,SPXW,O, , ,P,20160316,18350000,722,0,W
06:30:31.463,SPXW,O, , ,P,20160316,18200000,204,0,W
06:30:31.463,SPXW,O, , ,C,20160316,20450000,690,0,W
06:30:31.463,SPXW,O, , ,P,20160316,20250000,498,0,W
06:30:31.463,SPXW,O, , ,C,20160316,20050000,559,0,W
06:30:31.464,SPXW,O, , ,C,20160316,19900000,530,0,W
'''

quote_records = \
b'''10:19:12.950,BRKA,F,O, ,C,20160316,20450000,115,15500,W
10:19:13.140,BRKA,F,O, ,C,20160316,20300000,85,49000,W
10:19:13.140,BRKA,F,B, ,C,20160316,20300000,38,45000,W
10:19:13.140,BRKA,F,O, ,C,20160316,20350000,87,34000,W
10:19:13.140,BRKA,F,B, ,C,20160316,20350000,44,31000,W
10:19:13.140,BRKA,F,O, ,C,20160316,20100000,48,143000,W
10:19:13.140,BRKA,F,B, ,C,20160316,20100000,30,136000,W
10:19:13.141,BRKA,F,O, ,P,20160316,20150000,32,120000,W
10:19:13.141,BRKA,F,B, ,P,20160316,20150000,55,112000,W
10:19:13.141,BRKA,F,O, ,P,20160316,20050000,51,78000,W
10:19:13.141,BRKA,F,B, ,P,20160316,20050000,28,73000,W
10:19:13.141,BRKA,F,O, ,P,20160316,20250000,14,175000,W
10:19:13.141,BRKA,F,B, ,P,20160316,20250000,21,161000,W
10:19:13.141,BRKA,F,O, ,P,20160316,20200000,15,145000,W
10:19:13.141,BRKA,F,B, ,P,20160316,20200000,42,136000,W
10:20:13.152,BRKA,F,O, ,C,20160316,20050000,24,179000,W
10:20:13.152,BRKA,F,B, ,C,20160316,20450000,18,165000,W
10:20:13.152,BRKA,F,O, ,C,20160316,20300000,85,49000,W
10:20:13.152,BRKA,F,B, ,C,20160316,20300000,28,45000,W
10:20:13.160,BRKA,F,O, ,C,20160316,20100000,52,143000,W
10:20:13.160,BRKA,F,B, ,C,20160316,20100000,30,136000,W
10:20:13.161,BRKA,F,O, ,C,20160316,20300000,85,49000,W
10:20:13.161,BRKA,F,B, ,C,20160316,20300000,32,45000,W
10:20:13.161,BRKA,F,O, ,P,20160316,20050000,55,78000,W
10:20:13.161,BRKA,F,B, ,P,20160316,20050000,28,73000,W
10:20:13.161,BRKA,F,O, ,P,20160316,20100000,20,96000,W
10:20:13.161,BRKA,F,B, ,P,20160316,20100000,48,91000,W
10:20:13.161,BRKA,F,O, ,P,20160316,20150000,36,120000,W
10:20:13.161,BRKA,F,B, ,P,20160316,20150000,59,112000,W
'''

trade_records = \
b'''09:30:03.365,BRKA,T, , ,C,20160316,20200000,1,98000,W
09:30:03.481,BRKA,T, , ,P,20160316,20650000,2,442000,W
09:30:03.566,BRKA,T, ,L,P,20160316,20650000,1,6000,W
09:30:03.568,BRKA,T, ,L,P,20160316,20650000,2,13500,W
09:30:03.568,BRKA,T, ,L,P,20160316,19900000,1,34000,W
09:30:04.473,BRKA,T, ,L,C,20160316,19450000,15,714000,W
09:30:05.883,BRKA,T, ,L,C,20160316,20400000,4,24000,W
09:30:05.884,BRKA,T, ,L,P,20160316,20100000,10,87500,W
09:30:05.884,BRKA,T, ,L,P,20160316,20150000,10,109000,W
09:30:05.886,BRKA,T, ,L,C,20160316,20150000,3,119000,W
09:30:05.886,BRKA,T, ,L,C,20160316,20550000,3,7000,W
09:30:05.886,BRKA,T, ,L,P,20160316,19750000,3,15000,W
09:30:05.886,BRKA,T, ,L,P,20160316,20150000,3,110500,W
09:31:09.285,BRKA,T, ,L,P,20160316,20650000,10,26000,W
09:31:09.286,BRKA,T, ,L,P,20160316,20650000,10,33500,W
09:31:11.491,BRKA,T, , ,P,20160316,20650000,24,28000,W
09:31:11.586,BRKA,T, , ,P,20160316,19850000,12,28000,W
09:31:12.805,BRKA,T, , ,P,20160316,19800000,34,22000,W
09:31:12.831,BRKA,T, , ,P,20160316,19750000,44,17500,W
09:31:12.863,BRKA,T, , ,P,20160316,19750000,13,17500,W
09:31:13.640,BRKA,T, , ,P,20160316,19850000,1,28000,W
09:31:18.232,BRKA,T, ,L,C,20160316,20350000,26,34000,W
09:31:18.232,BRKA,T, ,L,C,20160316,20400000,26,23000,W
09:31:19.176,BRKA,T, ,L,P,20160316,19200000,3,3000,W
09:31:19.176,BRKA,T, ,L,P,20160316,19300000,3,3500,W
09:31:21.639,BRKA,T, ,L,C,20160316,20350000,4,34000,W
'''

other_records = \
b'''09:30:03.566,BRKA, , ,Sample Info,P,20160316,19500000,,,W
09:30:03.568,BRKA,X, ,Sample Info 2,P,20160316,19700000,,,W
09:30:05.886,BRKA,X, ,Sample Info 3,C,20160316,20550000,,,W
09:30:05.886,BRKA,X, ,Sample Info 4,P,20160316,19750000,,,W
09:30:05.886,BRKA,X, ,Sample Info 5,P,20160316,20150000,,,W
'''
if os.path.isdir('/tmp'):
    temp_dir = "/tmp/"
else:
    temp_dir =  tempfile.gettempdir()
    
import gzip
with gzip.open(temp_dir + '/BRKA_2018-01-01_data.gz', 'wb') as f:
    f.write(open_interest_records + quote_records + trade_records + other_records)

#cell 4
input_filename_provider = pq.PathFileNameProvider(temp_dir + '/BRKA_*.gz')
output_dir = temp_dir + '/pyqstrat'
if not os.path.isdir(output_dir): os.mkdir(output_dir)
output_file_prefix_mapper = pq.SingleDirectoryFileNameMapper(output_dir)

#cell 6
def is_trade(fields): return fields[2] == 'T'
def is_quote(fields) : return fields[2] in ['F', 'N'] # F is firm quote, N is no quote, i.e. either bid or offer is missing
def is_open_interest(fields):  return fields[2] == 'O'
def is_other(fields): return fields[2] == 'X'

# Keep only the lines that contain one of these substrings.  We can also use pq.RegExLineFilter but that is much slower than checking for substrings
line_filter = pq.SubStringLineFilter([",T,", ",F,", ",N,", ",O,", ",X,"])

#cell 8
def create_quote_parser(base_date, headers):
    timestamp_idx = headers.index('timestamp')
    qty_idx = headers.index('quantity')
    price_idx = headers.index('premium')
    bid_offer_idx = headers.index('side')
    # Id indices are used to uniquely identify an instrument.  Within this file, put/call, epxiry and strike uniquely identify an option
    id_indices = pq.get_field_indices(['putcall', 'expiration', 'strike'], headers)
    # Any other info besides qty, price, bid/offer and id that we want to store is stored in the meta field
    meta_indices = pq.get_field_indices(['info', 'exchange'], headers)
    # Prices in the input file are stored in thousands of cents so we divide them by 10000.0 to get dollars.
    # Bids are stored as "O" and offers are stored as "O"
    return pq.TextQuoteParser(is_quote, base_date, timestamp_idx, bid_offer_idx, price_idx, qty_idx, id_indices, meta_indices,
                                    pq.fast_time_milli_parser, "B", "O", 10000.0)
                                 
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
    return pq.TextOtherParser(is_other, base_date, timestamp_idx, id_indices, meta_indices, pq.fast_time_milli_parser)
    
def create_record_parser(quote_parser, trade_parser, open_interest_parser, other_parser):
    return pq.TextRecordParser(quote_parser, trade_parser, open_interest_parser, other_parser)
        

#cell 10
def create_quote_tob_aggregator(writer_creator, output_file_prefix, frequency = '1m'):
    return pq.QuoteTOBAggregator(writer_creator, output_file_prefix + ".tob", frequency = frequency)
                                 
def create_trade_bar_aggregator(writer_creator, output_file_prefix, frequency = '1m'):
    return pq.TradeBarAggregator(writer_creator, output_file_prefix + ".trades", frequency = frequency)
                                 
def create_other_aggregator(writer_creator, output_file_prefix):
    return pq.AllOtherAggregator(writer_creator, output_file_prefix + ".other")
                                 
def create_open_interest_aggregator(writer_creator, output_file_prefix):
    return pq.AllOpenInterestAggregator(writer_creator, output_file_prefix + ".open_interest")

#cell 13
def process(input_filename):
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
    # arrow writer creates an empty *.done marker file to indicate when its finished processing an input file, delete it so we can rerun
    done_file = output_dir + '/BRKA_2018-01-01_data.done'
    if os.path.exists(done_file): os.remove(done_file)
    pq.process_marketdata(input_filename_provider, process, num_processes = 8)

#cell 15
import pyarrow as pa

#cell 16
pa.RecordBatchFileReader(pa.OSFile(output_dir + '/BRKA_2018-01-01_data.open_interest.arrow', 'r')).read_pandas()

#cell 17
pa.RecordBatchFileReader(pa.OSFile(output_dir + '/BRKA_2018-01-01_data.trades.1m.arrow', 'r')).read_pandas()

#cell 18
pa.RecordBatchFileReader(pa.OSFile(output_dir + '/BRKA_2018-01-01_data.tob.1m.arrow', 'r')).read_pandas()

#cell 19
pa.RecordBatchFileReader(pa.OSFile(output_dir + '/BRKA_2018-01-01_data.other.arrow', 'r')).read_pandas()

