
# coding: utf-8

# In[2]:


import numpy as np
import pyarrow as pa
import os

def to_arrow_type(awtype):
    if awtype == ColumnType.INT32: return pa.int32()
    if awtype == ColumnType.INT64: return pa.int64()
    if awtype == ColumnType.FLOAT32: return pa.float32()
    if awtype == ColumnType.FLOAT64: return pa.float64()
    if awtype == ColumnType.STRING: return pa.string()
    if awtype == ColumnType.BOOL: return pa.bool_()
    if awtype == ColumnType.TIMESTAMP_MS: return pa.timestamp('ms')
    raise Exception(f'unknown type: {awtype}')
    
def to_numpy_type(awtype):
    if awtype == ColumnType.INT32: return np.int32
    if awtype == ColumnType.INT64: return np.int64
    if awtype == ColumnType.FLOAT32: return np.float32
    if awtype == ColumnType.FLOAT64: return np.float64
    if awtype == ColumnType.STRING: return np.object
    if awtype == ColumnType.BOOL: return np.bool8
    if awtype == ColumnType.TIMESTAMP_MS: return np.int64
    raise Exception(f'unknown type: {awtype}')
    
class ColumnType:
    BOOL = 0
    INT32 = 1
    INT64 = 2
    FLOAT32 = 3
    FLOAT64 = 4
    STRING = 5
    TIMESTAMP_MS = 6
    
def open_arrow_file(file_name, schema):
    output_file = pa.OSFile(file_name, 'wb')
    return output_file, pa.RecordBatchFileWriter(output_file, schema)

class ArrowWriter:
    def __init__(self, output_file_prefix, schema, create_batch_id_file, max_batch_size):
        self.output_file_prefix = output_file_prefix
        self.max_batch_size = max_batch_size
        self.create_batch_id_file = create_batch_id_file
        fields = [pa.field('line_num', pa.int64())]
        if max_batch_size == np.inf: max_batch_size = int(1e6)
        self.line_num = np.empty(max_batch_size, dtype = np.int64)
        self.arrays = []
        for tup in schema:
            fields.append(pa.field(tup[0], to_arrow_type(tup[1])))
            self.arrays.append(np.empty(max_batch_size, dtype = to_numpy_type(tup[1])))
        self.schema = pa.schema(fields)
        self.output_file, self.writer = open_arrow_file(output_file_prefix + ".arrow.tmp", self.schema)
        if self.create_batch_id_file:
            self.id_schema = pa.schema([pa.field('id', pa.string()), pa.field('batch_id', pa.int64())])
            self.id_file, self.id_writer = open_arrow_file(output_file_prefix + '.batch_ids.arrow.tmp', self.id_schema)
            self.batch_ids = np.empty(int(1e6), dtype = np.object)
            self.batch_num = 0
        self.closed = False
        self.record_num = 0
        
    def add_record(self, record, line_number):
        if self.record_num == len(self.arrays[0]): any(map(lambda a : a.resize(len(a) * 1.25), self.arrays))
        for i, val in enumerate(record):
            self.arrays[i][self.record_num] = val
        if self.record_num == len(self.line_num): self.line_num.resize(len(self.line_num) * 1.25)
        self.line_num[self.record_num] = line_number
        self.record_num += 1
        if self.record_num == self.max_batch_size:
            self.write_batch()
        
    def write_batch(self, batch_id = None):
        batch = pa.RecordBatch.from_arrays([pa.array(self.line_num)] + [pa.array(array) for array in self.arrays], self.schema)
        
        if self.record_num < self.max_batch_size:
            batch = batch.slice(length = self.record_num)
        self.writer.write_batch(batch)
        
        if self.create_batch_id_file:
            if batch_id is None: raise Exception('batch id must be provided if create_batch_id_file was set')
            if self.batch_num == len(self.batch_ids): self.batch_ids.resize(len(self.batch_ids) * 1.25)
            self.batch_ids[self.batch_num] = batch_id
            self.batch_num += 1
        self.record_num = 0
        
    def close(self, success = True):
        if self.closed: return
        if (self.record_num > 0): self.write_batch()
        self.writer.close()
        self.output_file.close()
        
        if self.create_batch_id_file:
            print(f'batch num = {self.batch_num} batch_ids = {self.batch_ids}')
            print(self.batch_ids[:self.batch_num])
            print(np.arange(self.batch_num))
            id_batch = pa.RecordBatch.from_arrays([pa.array(self.batch_ids[:self.batch_num]), pa.array(np.arange(self.batch_num))], self.id_schema)
            self.id_writer.write_batch(id_batch)
            
        if success and os.path.isfile(self.output_file_prefix + '.arrow.tmp'):
            os.rename(self.output_file_prefix + '.arrow.tmp', self.output_file_prefix + '.arrow')
            
        if self.create_batch_id_file:
            if success and os.path.isfile(self.output_file_prefix + '.batch_ids.arrow.tmp'):
                os.rename(self.output_file_prefix + '.batch_ids.arrow.tmp', self.output_file_prefix + '.batch_ids.arrow')
            self.id_writer.close()
            self.id_file.close()

        self.closed = True
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        success = (exc_type is None)
        self.close(success)
        
def write_file(file_prefix):
    a = ArrowWriter(file_prefix, schema = [('a', ColumnType.INT64), ('b', ColumnType.INT64)], create_batch_id_file = True, max_batch_size=10)
    a.add_record((35, 22), 1)
    a.close()
    return file_prefix
    
if __name__ == "__main__":
    import concurrent
    files = ['/tmp/x', '/tmp/y']
    futs = []
    with concurrent.futures.ProcessPoolExecutor(2) as executor:
        for file in files:
            fut = executor.submit(write_file, file)
            futs.append(fut)

        for future in concurrent.futures.as_completed(futs):
            filename = future.result()
            print(f'completed: {filename}')
            
    reader = pa.open_file(pa.OSFile('/tmp/x.arrow'))
    assert(reader.num_record_batches == 1)
    df = reader.read_pandas()
    assert(len(df) == 1)

