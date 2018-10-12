
# coding: utf-8

# In[1]:


import glob
import os
import re
import datetime
import dateutil
import numpy as np
import concurrent
import pyarrow as pa
import pathlib
from timeit import default_timer as timer
from pyqstrat import *
from pyqstrat.pq_utils import *
#from pyqstrat.pyqstrat_cpp import *


# In[2]:


VERBOSE = False

class PathFileNameProvider:
    def __init__(self, path, include_pattern = None, exclude_pattern = None):
        self.path = path
        self.include_pattern = include_pattern
        self.exclude_pattern = exclude_pattern
        
    def __call__(self):
        files = glob.glob(self.path)
        if not len(files):
            raise Exception(f'no matching files found with pattern: {self.path}')
        if self.include_pattern:
            files = [file for file in files if self.include_pattern in file]
        if self.exclude_pattern:
            files = [file for file in files if self.exclude_pattern not in file]
        if not len(files):
            raise Exception(f'no matching files for: {self.path} including: {self.include_pattern} excluding: {self.exclude_pattern}')
        return files
    
class SingleDirectoryFileNameMapper:
    def __init__(self, output_dir):
        if not os.path.isdir(output_dir): raise Exception(f'{output_dir} does not exist')
        self.output_dir = output_dir

    def __call__(self, input_filepath):
        if self.output_dir is None:
            dirname = os.path.dirname(input_filepath)
            dirname = os.path.join(dirname, 'output')
        else:
            dirname = self.output_dir
            
        if not os.path.isdir(dirname): raise Exception(f'{dirname} does not exist')
     
        input_filename = os.path.basename(input_filepath)
        exts = '\.txt$|\.gz$|\.bzip2$|\.bz$|\.tar$|\.zip$|\.csv$'
        while (re.search(exts, input_filename)):
            input_filename = '.'.join(input_filename.split('.')[:-1])
            if VERBOSE: print(f'got input file: {input_filename}')
        output_prefix = os.path.join(dirname, input_filename)
        return output_prefix

class TextHeaderParser:
    def __init__(self, record_generator, skip_rows = 0, separator = ',', make_lowercase = True):
        self.record_generator = record_generator
        self.skip_rows = 0
        self.separator = separator
        self.make_lowercase = make_lowercase
        
    def __call__(self, input_filename, compression):
        with self.record_generator(input_filename, compression) as f:
            headers = None
            for line_num, line in enumerate(f):
                line = line.decode()
                if line_num < self.skip_rows: continue
                headers = line.split(self.separator)
                headers = [re.sub('[^A-Za-z0-9 ]+', '', header) for header in headers]
                if len(headers) == 1:
                    raise Exception(f'Could not parse headers: {line} with separator: {self.separator}')
                break

            if headers is None: raise Exception('no headers found')
            if self.make_lowercase: headers = [header.lower() for header in headers]
            if VERBOSE: print(f'Found headers: {headers}')
            return headers
    
def infer_compression(input_filename):
    parts = input_filename.split('.')
    if len(parts) <= 1: return None
    suffix = parts[-1]
    if suffix == 'gz': return 'gzip'
    if suffix == 'bz2': return 'bz2'
    if suffix =='zip': return 'zip'
    if suffix == 'xz': return 'xz'
    return None

def text_file_record_generator(filename, compression):
    if compression is None: compression = infer_compression(self.filename)
    if compression == None or compression == '':
        return open(filename, 'r')
    if compression == 'gzip':
        import gzip
        return gzip.open(filename, 'r')
    if compression == 'bz2':
        import bz2
        return bz2.BZ2File(filename, 'r')
    if compression == 'zip':
        import zipfile
        zf = zipfile.ZipFile(filename, mode, zipfile.ZIP_DEFLATED)
        zip_names = zf.namelist()
        if len(zip_names) == 0: raise ValueError(f'zero files found in ZIP file {filename}')
        if len(zip_names) > 1: raise ValueError(f'{len(zip_names)} files found in ZIP file {filename} {zip_names}.  Only 1 allowed')
        return zf.open(zip_names.pop())
    if compression == 'xz':
        import_lzma
        return lzma.LZMAFile(filename, 'r')
    raise ValueError(f'unrecognized compression: {compression} for file: {filename}')
    
def get_field_indices(field_names, headers):
    field_ids = np.ones(len(field_names), dtype = np.int) * -1
    for i, field_name in enumerate(field_names):
        if field_name not in headers: raise ParseException(f'{field_name} not in {headers}')
        field_ids[i] = headers.index(field_name)
    return field_ids

def _run_multi_process(func, raise_on_error):
    fut_map = {}
    with concurrent.futures.ProcessPoolExecutor(self.max_processes) as executor:
        for suggestion in self.generator:
            if suggestion is None: continue
            future = executor.submit(self.cost_func, suggestion)
            fut_map[future] = suggestion

        for future in concurrent.futures.as_completed(fut_map):
            try:
                cost, other_costs = future.result()
            except Exception as e:
                new_exc = type(e)(f'Exception: {str(e)} with suggestion: {suggestion}').with_traceback(sys.exc_info()[2])
                if raise_on_error: raise new_exc
                else: print(str(new_exc))
                continue
            suggestion = fut_map[future]
            self.experiments.append(Experiment(suggestion, cost, other_costs))
            
def base_date_filename_mapper(input_file_path):
    filename = os.path.basename(input_file_path)
    base_date = dateutil.parser.parse(filename, fuzzy=True)
    return round(millis_since_epoch(base_date))

def arrow_writer_creator(output_file_prefix, schema, create_batch_id_file, batch_size):
    return ArrowWriter(output_file_prefix, schema, create_batch_id_file, batch_size)

def create_text_file_processor(record_generator, line_filter, record_parser, bad_line_handler, record_filter, missing_data_handler,
                                quote_aggregator, trade_aggregator, open_interest_aggregator, other_aggregator, skip_rows = 1): 
    return TextFileProcessor(record_generator,
                                line_filter,
                                record_parser,
                                bad_line_handler,
                                record_filter,
                                missing_data_handler,
                                quote_aggregator,
                                trade_aggregator,
                                open_interest_aggregator,
                                other_aggregator,
                                skip_rows)

def process_marketdata_file(input_filename,
                 output_file_prefix_mapper,
                 quote_parser_creator,
                 trade_parser_creator,
                 open_interest_parser_creator,
                 other_parser_creator,
                 record_parser_creator,
                 quote_aggregator_creator,
                 trade_aggregator_creator,
                 open_interest_aggregator_creator,
                 other_aggregator_creator,
                 line_filter = None, 
                 compression = None,
                 base_date_mapper = base_date_filename_mapper,
                 file_processor_creator = create_text_file_processor,
                 header_parser_creator = lambda record_generator :  TextHeaderParser(record_generator),
                 header_record_generator = text_file_record_generator,
                 record_generator = text_file_decompressor,
                 bad_line_handler = PrintBadLineHandler(),
                 record_filter = None,
                 missing_data_handler = price_qty_missing_data_handler, 
                 writer_creator = arrow_writer_creator):
    
    output_file_prefix = output_file_prefix_mapper(input_filename)
    base_date = base_date_mapper(input_filename)
    
    if not is_newer(input_filename, output_file_prefix + '.done'):
        print(f'{output_file_prefix + ".done"} exists and is not older than: {input_filename}')
        return
                                
    header_parser = header_parser_creator(header_record_generator)
    print(f'starting file: {input_filename}')
    if compression is None: compression = infer_compression(input_filename)
    headers = header_parser(input_filename, compression)
    
    quote_parser = None
    if quote_parser_creator : quote_parser = quote_parser_creator(base_date, headers)

    trade_parser = None
    if trade_parser_creator: trade_parser = trade_parser_creator(base_date, headers)

    open_interest_parser = None
    if open_interest_parser_creator: open_interest_parser = open_interest_parser_creator(base_date, headers)

    other_parser = None
    if other_parser_creator: other_parser = other_parser_creator(base_date, headers)

    record_parser = record_parser_creator(quote_parser, trade_parser, open_interest_parser, other_parser)

    quote_aggregator = quote_aggregator_creator(writer_creator, output_file_prefix)
    trade_aggregator = trade_aggregator_creator(writer_creator, output_file_prefix)
    open_interest_aggregator = open_interest_aggregator_creator(writer_creator, output_file_prefix)
    other_aggregator = other_aggregator_creator(writer_creator, output_file_prefix)

    file_processor = file_processor_creator(
        record_generator, 
        line_filter, 
        record_parser, 
        bad_line_handler, 
        record_filter, 
        missing_data_handler,
        quote_aggregator, 
        trade_aggregator, 
        open_interest_aggregator, 
        other_aggregator
    )

    start = timer()
    lines_processed = file_processor(input_filename, compression)
    end = timer()
    duration = round((end - start) * 1000)
    touch(output_file_prefix + '.done')
    print(f"processed {input_filename} {lines_processed} lines in {duration} milliseconds")
                    
def process_marketdata(input_filename_provider, file_processor, num_processes, raise_on_error = True):
    input_filenames = input_filename_provider()
    with concurrent.futures.ProcessPoolExecutor(num_processes) as executor:
        fut_filename_map = {}
        for input_filename in input_filenames:
            fut = executor.submit(file_processor, input_filename)
            fut_filename_map[fut] = input_filename
        for fut in concurrent.futures.as_completed(fut_filename_map):
            try:
                fut.result()
                if VERBOSE: print(f'done filename: {fut_filename_map[fut]}')
            except Exception as e:
                new_exc = type(e)(f'Exception: {str(e)}').with_traceback(sys.exc_info()[2])
                if raise_on_error: 
                    raise new_exc
                else: 
                    print(str(new_exc))
                    continue
            

