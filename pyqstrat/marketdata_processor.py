#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import os
import sys
import re
import datetime
import dateutil
import numpy as np
import concurrent
import pyarrow as pa
import pathlib
from timeit import default_timer as timer

from pyqstrat import *


# In[ ]:


VERBOSE = False

class PathFileNameProvider:
    """A helper class that, given a pattern such as such as "/tmp/abc*.gz" and an optional include and exclude pattern, 
    returns names of all files that match
    """
    def __init__(self, path, include_pattern = None, exclude_pattern = None):
        '''
        Args:
            path (str): A pattern such as "/tmp/abc*.gz"
            include_pattern (str): Given a pattern such as "xzy", will return only filenames that contain xyz
            exclude_pattern (str): Given a pattern such as "_tmp", will exclude all filenames containing _tmp
        '''
        self.path = path
        self.include_pattern = include_pattern
        self.exclude_pattern = exclude_pattern
        
    def __call__(self):
        """
        Returns:
            list of str: all matching filenames
        """
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
    """A helper class that provides a mapping from input filenames to their corresponding output filenames in an output directory."""
    def __init__(self, output_dir):
        """
        Args:
            output_dir (str): The directory where we want to write output files
        """
        if not os.path.isdir(output_dir): raise Exception(f'{output_dir} does not exist')
        self.output_dir = output_dir

    def __call__(self, input_filepath):
        """
        Args:
            input_filepath (str): The input file that we are creating an output file for, e.g. "/home/xzy.gz"
        Returns:
            str: Output file path for that input.  We take the filename from the input filepath, strip out any extension 
                and prepend the output directory name
        """
        
        if self.output_dir is None:
            dirname = os.path.dirname(input_filepath)
            dirname = os.path.join(dirname, 'output')
        else:
            dirname = self.output_dir
            
        if not os.path.isdir(dirname): raise Exception(f'{dirname} does not exist')
     
        input_filename = os.path.basename(input_filepath)
        exts = r'\.txt$|\.gz$|\.bzip2$|\.bz$|\.tar$|\.zip$|\.csv$'
        while (re.search(exts, input_filename)):
            input_filename = '.'.join(input_filename.split('.')[:-1])
            if VERBOSE: print(f'got input file: {input_filename}')
        output_prefix = os.path.join(dirname, input_filename)
        return output_prefix

class TextHeaderParser:
    """
    Parses column headers from a text file containing market data
    """
    def __init__(self, record_generator, skip_rows = 0, separator = ',', make_lowercase = True):
        """
        Args:
        
            record_generator: A function that takes a filename and its compression type and returns an object
                that we can use to iterate through lines in that file
            skip_rows (int, optional): Number of rows to skip before starting to read the file.  Default is 0
            separator (str, optional): Separator for headers.  Defaults to ,
            make_lowercase (bool, optional): Whether to convert headers to lowercase before returning them
        """
        self.record_generator = record_generator
        self.skip_rows = 0
        self.separator = separator
        self.make_lowercase = make_lowercase
        
    def __call__(self, input_filename, compression):
        """
        Args:
        
        input_filename (str): The file to read
        compression (str): Compression type, e.g. "gzip", or None if the file is not compressed
        
        Returns:
            list of str: column headers
        """
        
        decode_needed = (compression is not None and compression != "")
        
        with self.record_generator(input_filename, compression) as f:
            headers = None
            for line_num, line in enumerate(f):
                if decode_needed: line = line.decode()
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
        
            parts = input_filename.split('.')
 
def text_file_record_generator(filename, compression):
    """
    A helper function that returns an object that we can use to iterate through lines in the input file
    Args:
        filename (str): The input filename
        compression (str): The compression type of the input file or None if its not compressed    
    """
    if compression is None: compression = infer_compression(filename)
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

def base_date_filename_mapper(input_file_path):
    """
    A helper function that parses out the date from a filename.  For example, given a file such as "/tmp/spx_2018-08-09", this parses out the 
    date part of the filename and returns milliseconds (no fractions) since the epoch to that date.
    
    Args:
    
        input_filepath (str): Full path to the input file
    
    Returns:
    
       int: Milliseconds since unix epoch to the date implied by that file
    
    >>> base_date_filename_mapper("/tmp/spy_1970-1-2_quotes.gz")
    86400000
    """
    filename = os.path.basename(input_file_path)
    base_date = dateutil.parser.parse(filename, fuzzy=True)
    return round(millis_since_epoch(base_date))

def create_text_file_processor(record_generator, line_filter, record_parser, bad_line_handler, record_filter, missing_data_handler,
                               aggregators, skip_rows = 1): 
    return TextFileProcessor(record_generator,
                                line_filter,
                                record_parser,
                                bad_line_handler,
                                record_filter,
                                missing_data_handler,
                                aggregators,
                                skip_rows)

def get_field_indices(field_names, headers):
    """
    Helper function to get indices of field names in a list of headers
    
    Args:
        field_names (list of str): The fields we want indices of
        headers (list of str): All headers
        
    Returns:
        list of int: indices of each field name in the headers list
    """
    field_ids = np.ones(len(field_names), dtype = np.int) * -1
    for i, field_name in enumerate(field_names):
        if field_name not in headers: raise ParseException(f'{field_name} not in {headers}')
        field_ids[i] = headers.index(field_name)
    return field_ids

def process_marketdata_file(input_filename,
                 output_file_prefix_mapper,
                 record_parser_creator,
                 aggregator_creator,
                 line_filter = None, 
                 compression = None,
                 base_date_mapper = None,
                 file_processor_creator = create_text_file_processor,
                 header_parser_creator = lambda record_generator :  TextHeaderParser(record_generator),
                 header_record_generator = text_file_record_generator,
                 record_generator = TextFileDecompressor(),
                 bad_line_handler = PrintBadLineHandler(),
                 record_filter = None,
                 missing_data_handler = PriceQtyMissingDataHandler(), 
                 writer_creator = ArrowWriterCreator()):
    
    """
    Processes a single market data file
    
    Args:
        input_filename (str):
        output_file_prefix_mapper: A function that takes an input filename and returns the corresponding output filename we want
        record_parser_creator:  A function that takes a date and a list of column names and returns a 
            function that can take a list of fields and return a subclass of Record
        line_filter (optional): A function that takes a line and decides whether we want to keep it or discard it.  Defaults to None
        compression (str, optional): Compression type for the input file.  Defaults to None
        base_date_mapper (optional): A function that takes an input filename and returns the date implied by the filename, 
            represented as millis since epoch.  Defaults to helper :obj:`function base_date_filename_mapper`
        file_processor_creator (optional): A function that returns an object that we can use to iterate through lines in a file.  Defaults to
            helper function :obj:`create_text_file_processor`
        bad_line_handler (optional): A function that takes a line that we could not parse, and either parses it or does something else
            like recording debugging info, or stopping the processing by raising an exception.  Defaults to helper function 
            :obj:`PrintBadLineHandler`
        record_filter (optional): A function that takes a parsed TradeRecord, QuoteRecord, OpenInterestRecord or OtherRecord and decides whether we
            want to keep it or discard it.  Defaults to None
        missing_data_handler (optional):  A function that takes a parsed TradeRecord, QuoteRecord, OpenInterestRecord or OtherRecord, and decides
            deals with any data that is missing in those records.  For example, 0 for bid could be replaced by NAN.  Defaults to helper function:
            :obj:`price_qty_missing_data_handler`
        writer_creator (optional): A function that takes an output_file_prefix, schema, whether to create a batch id file, and batch_size
            and returns a subclass of :obj:`Writer`.  Defaults to helper function: :obj:`arrow_writer_creator`
    """
    
    output_file_prefix = output_file_prefix_mapper(input_filename)
    
    base_date = 0
    
    if base_date_mapper is not None: base_date = base_date_mapper(input_filename)
    
    if not is_newer(input_filename, output_file_prefix + '.done'):
        print(f'{output_file_prefix + ".done"} exists and is not older than: {input_filename}')
        return
                                
    header_parser = header_parser_creator(header_record_generator)
    print(f'starting file: {input_filename}')
    if compression is None: compression = infer_compression(input_filename)
    headers = header_parser(input_filename, compression)
    
    record_parser = record_parser_creator(base_date, headers)
    
    aggregators = aggregator_creator(writer_creator, output_file_prefix)

    file_processor = file_processor_creator(
        record_generator, 
        line_filter, 
        record_parser, 
        bad_line_handler, 
        record_filter, 
        missing_data_handler,
        aggregators
    )

    start = timer()
    if compression is None: compression = ""
    lines_processed = file_processor(input_filename, compression)
    end = timer()
    duration = round((end - start) * 1000)
    touch(output_file_prefix + '.done')
    print(f"processed: {input_filename} {lines_processed} lines in {duration} milliseconds")
                    
def process_marketdata(input_filename_provider, file_processor, num_processes = None, raise_on_error = True):
    """
    Top level function to process a set of market data files
    
    Args:
        input_filename_provider: A function that returns a list of filenames (incl path) we need to process.
        file_processor: A function that takes an input filename and processes it, returning number of lines processed. 
        num_processes (int, optional): The number of processes to run to parse these files.  If set to None, we use the number of cores
            present on your machine.  Defaults to None
        raise_on_error (bool, optional): If set, we raise an exception when there is a problem with parsing a file, so we can see a stack
            trace and diagnose the problem.  If not set, we print the error and continue.  Defaults to True
    """
    
    input_filenames = input_filename_provider()
    if sys.platform in ["win32", "cygwin"] and num_processes > 1:
        raise Exception("num_processes > 1 not supported on windows")
     
    if num_processes == 1 or sys.platform in ["win32", "cygwin"]:
        for input_filename in input_filenames:
            try:
                file_processor(input_filename)
            except Exception as e:
                new_exc = type(e)(f'Exception: {str(e)}').with_traceback(sys.exc_info()[2])
                if raise_on_error: 
                    raise new_exc
                else: 
                    print(str(new_exc))
                    continue
    else:
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


# In[ ]:




