//
//  read_file.cpp
//  py_c_test
//
//  Created by Sal Abbasi on 9/11/22.
//
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API 1
#include <Python.h>
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <time.h>
#include "structmember.h"
#include "csv_reader.hpp"
#include "numpy/ndarrayobject.h"


#include <time.h>
#include <iomanip>
#include <sstream>


//strptime not inmplemented in windows
#ifdef _MSC_VER
extern "C" char* strptime(const char* s,
                          const char* f,
                          struct tm* tm) {
  std::istringstream input(s);
  input.imbue(std::locale(setlocale(LC_ALL, nullptr)));
  input >> std::get_time(tm, f);
  if (input.fail()) {
    return nullptr;
  }
  return (char*)(s + input.tellg());
}
#endif


using namespace std;

static int read_list(PyObject* list, vector<int>& vec) {
    if (list == NULL) return 0; // Empty vector
    Py_ssize_t n = PyList_Size(list);
    for (int i=0; i < n; i++) {
        PyObject* item = PyList_GetItem(list, i);
        if (!PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "list items must be integers.");
            return 0;
        }
        int elem = static_cast<int>(PyLong_AsLong(item));
        vec.push_back(elem);
    }
    return -1;
}

static int read_list(PyObject* list, vector<string>& vec) {
    if (list == NULL) return 0; // Empty vector
    Py_ssize_t n = PyList_Size(list);
    for (int i=0; i < n; i++) {
        PyObject* item = PyList_GetItem(list, i);
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "list items must be strings.");
            return 0;
        }
        PyObject* ascii = PyUnicode_AsASCIIString(item);
        char* ret_string = PyBytes_AsString(ascii);
        vec.push_back(std::string(ret_string));
        Py_DECREF(ascii);
    }
    return -1;
}

static PyObject* create_np_str_array(const std::vector<std::string>& vals, size_t itemsize){
    
    size_t mem_size = vals.size() * itemsize;
    
    void * mem = PyDataMem_NEW(mem_size);
    
    size_t cur_index=0;
    
    for (const auto& val : vals){
        for(size_t i = 0; i < itemsize; i++){
            char ch = i < val.size() ? val[i] : 0; // fill with NULL if string too short
            reinterpret_cast<char*>(mem)[cur_index] = ch;
            cur_index++;
        }
    }

    npy_intp dim = static_cast<npy_intp>(vals.size());
    
    PyObject* arr = PyArray_New(&PyArray_Type, 1, &dim, NPY_STRING, NULL, mem,
                                static_cast<int>(itemsize), NPY_ARRAY_CARRAY | NPY_ARRAY_OWNDATA, NULL);
    PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
    return arr;
}

template<typename T> PyObject* create_np_array(PyArray_Descr* descr, void* data) {
    npy_intp dims[1];
    auto vec = static_cast<vector<T>*>(data);
    dims[0] = vec->size();
    auto _data = new T[vec->size()];
    ::memcpy(
      _data,
      vec->data(),
      vec->size() * sizeof(T));
    delete vec;

    
    PyObject* arr = PyArray_NewFromDescr(&PyArray_Type, descr, 1, dims, NULL, _data,
                                         NPY_ARRAY_CARRAY | NPY_ARRAY_OWNDATA , NULL);
    PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);

    if (arr == NULL) {
        PyErr_SetString(PyExc_TypeError, "could not allocate numpy array");
        return NULL;
    }
    return arr;
}

                                  
static PyObject* create_np_array(const std::string& dtype, void* data) {
    
    PyObject* _dtype = Py_BuildValue("s", dtype.c_str());
    PyArray_Descr* descr;
    PyArray_DescrConverter(_dtype, &descr);
    Py_XDECREF(_dtype);
    
    PyObject* arr = NULL;
    if (dtype[0] == 'S') {
        size_t itemsize = atoi(dtype.substr(1).c_str());
        if (itemsize <= 0) {
            PyErr_SetString(PyExc_TypeError, "item size must be a positive int");
            return NULL;
        }
        auto col = static_cast<vector<string>*>(data);
        arr = create_np_str_array(*col, itemsize);
        delete col;
    } else if (dtype.substr(0, 3) == "M8[") {
        arr = create_np_array<int64_t>(descr, data);
    } else if (dtype == "i1") {
        arr = create_np_array<int8_t>(descr, data);
    } else if (dtype == "i4") {
        arr = create_np_array<int32_t>(descr, data);
    } else if (dtype == "i8") {
        arr = create_np_array<int64_t>(descr, data);
    } else if (dtype == "f4") {
        arr = create_np_array<float>(descr, data);
    } else if (dtype == "f8") {
        arr = create_np_array<double>(descr, data);
    } else {
        PyErr_SetString(PyExc_TypeError, "only f4, f8, i1, i4, i8, M8[*] and S[n] datatypes are supported");
    }
    return arr;
}

static PyObject*
read_file(PyObject* self, PyObject* args, PyObject* kwargs) {
    char* filename = NULL;
    PyObject* _col_indices = NULL;
    PyObject* _dtypes = NULL;
    char* separator = NULL;
    int skip_rows = 1;
    int max_rows = 0;
    
    const char *kwlist[] = {
        "filename",
        "col_indices",
        "dtypes",
        "separator",
        "skip_rows",
        "max_rows",
        NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "sOO|siOi",
                                     const_cast<char**>(kwlist),
                                     &filename,
                                     &_col_indices,
                                     &_dtypes,
                                     &separator,
                                     &skip_rows,
                                     &max_rows)) {
        return NULL;
    }
    if (!PyList_Check(_col_indices)) {
        PyErr_SetString(PyExc_RuntimeError, "col_indices must be a list");
        return NULL;
    }

    if (!PyList_Check(_dtypes)) {
        PyErr_SetString(PyExc_RuntimeError, "dtypes must be a list");
        return NULL;
    }
    vector<string> dtypes;
    vector<int> col_indices;
    if (!read_list(_col_indices, col_indices)) return NULL;
    
    int max_col_idx = -1;
    for (auto i: col_indices) {
        if (i <= max_col_idx) {
            PyErr_SetString(PyExc_RuntimeError, "col_indices must be monotonically increasing");
            return NULL;
        }
        max_col_idx = i;
    }
    
    if (!read_list(_dtypes, dtypes)) return NULL;
    if (skip_rows < 0) {
        PyErr_SetString(PyExc_RuntimeError, "skip_rows must be >= 0");
        return NULL;
    }
    if (max_rows < 0) {
        PyErr_SetString(PyExc_RuntimeError, "max_rows must be positive (or zero to read all rows)");
        return NULL;
    }
    
    if (col_indices.size() != dtypes.size()) {
        PyErr_SetString(PyExc_RuntimeError, "col_indices and dtypes must be same size");
        return NULL;
    }
    
    // release the gil
    PyThreadState *_save;
    _save = PyEval_SaveThread();
    vector<void*> data;
    try {
        read_csv(filename, col_indices, dtypes, separator[0], skip_rows, max_rows, data);
    } catch (const std::exception& ex) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return NULL;
    }
    //reaquire the gil
    PyEval_RestoreThread(_save);
    
    auto arrays = PyList_New(dtypes.size());

    int i = 0;
    for (auto _dtype: dtypes) {
        PyObject* arr = create_np_array(_dtype, data[i]);
        if (arr == NULL) {
            Py_XDECREF(arrays);
            return NULL;
        }
        PyList_SetItem(arrays, i, arr);
        i++;
    }
    return arrays;
}

static time_t time_to_epoch ( const struct tm *ltm, int utcdiff ) {
   const int mon_days [] =
      {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
   long tyears, tdays, leaps, utc_hrs;
   int i;

   tyears = ltm->tm_year - 70 ; // tm->tm_year is from 1900.
   leaps = (tyears + 2) / 4; // no of next two lines until year 2100.
   //i = (ltm->tm_year – 100) / 100;
   //leaps -= ( (i/4)*3 + i%4 );
   tdays = 0;
   for (i=0; i < ltm->tm_mon; i++) tdays += mon_days[i];

   tdays += ltm->tm_mday-1; // days of month passed.
   tdays = tdays + (tyears * 365) + leaps;

   utc_hrs = ltm->tm_hour + utcdiff; // for your time zone.
   return (tdays * 86400) + (utc_hrs * 3600) + (ltm->tm_min * 60) + ltm->tm_sec;
}

static PyObject*
parse_datetimes(PyObject* self, PyObject* args) {
    PyObject* datetimes = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &datetimes))  return NULL;

    if (datetimes == NULL) return NULL;

    npy_intp n = PyArray_SIZE(datetimes);
    vector<::time_t> output(n);

    struct tm tm;
    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
        PyObject *item = PySequence_GetItem(datetimes, i);
        if (!item) {
            return NULL;
        }
        Py_ssize_t size;
        const char *time_str = PyUnicode_AsUTF8AndSize(item, &size);
        if (!time_str) {
            return NULL;
        }
        if (time_str == NULL) return NULL;
        ::memset(&tm, 0, sizeof(tm));
        ::strptime(time_str, "%Y-%m-%dT%H:%M:%S", &tm);
        ::time_t event_time = ::time_to_epoch(&tm, 0);
        output[i] = event_time;
    }
    PyObject* arr = create_np_array("M8[s]", &output);
    return arr;
}

    
static PyMethodDef IOModuleMethods[] = {
    {"read_file", (PyCFunction)(void(*)(void))read_file, METH_VARARGS | METH_KEYWORDS, "read a file"},
    {"parse_datetimes", parse_datetimes, METH_VARARGS, "parse datetimes"},
    {NULL, NULL, 0, NULL}
};

    
static struct PyModuleDef io_module = {
    PyModuleDef_HEAD_INIT,
    "pyqstrat_io",
    NULL,
    -1,
    IOModuleMethods
};

/* The classes below are exported */
#ifdef __GNUC__
#pragma GCC visibility push(default)
#endif

PyMODINIT_FUNC
PyInit_pyqstrat_io(void) {
    import_array();
    return PyModule_Create(&io_module);
}

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

