#include "py_import_call_execute.hpp"

#include "Python.h"

#include <string>
#include <vector>

std::vector<std::string> tokenize2(const char* str, const char separator) {
    std::vector<std::string> tokens;
    const char* token_start = str;
    while (true) {
        if ((*str == separator) || (*str == '\0')) {
            tokens.push_back(std::string(token_start, str));
            token_start = str + 1;
            if (*str == '\0') break;
        }
        str++;
    }
    return tokens;
}


#ifdef __cplusplus
extern "C" {
#endif

/** Takes a path and adds it to sys.paths by calling PyRun_SimpleString.
 * This does rather laborious C string concatenation so that it will work in
 * a primitive C environment.
 *
 * Returns 0 on success, non-zero on failure.
 */
int add_path_to_sys_module(const char *path) {
    int ret = 0;
    const char *prefix = "import sys\nsys.path.append(\"";
    const char *suffix = "\")\n";
    char *command = (char*)malloc(strlen(prefix)
                                  + strlen(path)
                                  + strlen(suffix)
                                  + 1);
    if (! command) {
        return -1;
    }
    strcpy(command, prefix);
    strcat(command, path);
    strcat(command, suffix);
    ret = PyRun_SimpleString(command);
#ifdef DEBUG
    printf("Calling PyRun_SimpleString() with:\n");
    printf("%s", command);
    printf("PyRun_SimpleString() returned: %d\n", ret);
    fflush(stdout);
#endif
    free(command);
    return ret;
}
    


/** This imports a Python module and calls a specific function in it.
 * It's arguments are similar to main():
 * argc - Number of strings in argv
 * argv - Expected to be 4 strings:
 *      - Name of the executable.
 *      - Path to the directory that the Python module is in.
 *      - Name of the Python module.
 *      - Name of the function in the module.
 *
 * The Python interpreter will be initialised and the path to the Python module
 * will be added to sys.paths then the module will be imported.
 * The function will be called with no arguments and its return value will be
 * ignored.
 *
 * This returns 0 on success, non-zero on failure.
 */
int import_call_execute(int argc, const char *argv[]) {
    int return_value = 0;
    PyObject *pModule   = NULL;
    PyObject *pFunc     = NULL;
    PyObject *pResult   = NULL;

    if (argc != 4) {
        fprintf(stderr,
                "Wrong arguments!"
                " Usage: %s package_path module function\n", argv[0]);
        return_value = -1;
        goto except;
    }
    Py_SetProgramName((wchar_t*)argv[0]);
    Py_Initialize();
    
    for (auto dir : tokenize2(argv[1], ':')) {
        if (add_path_to_sys_module(dir.c_str())) {
            return_value = -2;
            goto except;
        }
    }
    
    fprintf(stdout, "running %s\n", argv[2]);
    
    pModule = PyImport_ImportModule(argv[2]);
    if (! pModule) {
        fprintf(stderr,
                "%s: Failed to load module \"%s\"\n", argv[0], argv[2]);
        return_value = -3;
        goto except;
    }
    pFunc = PyObject_GetAttrString(pModule, argv[3]);
    if (! pFunc) {
        fprintf(stderr,
                "%s: Can not find function \"%s\"\n", argv[0], argv[3]);
        return_value = -4;
        goto except;
    }
    if (! PyCallable_Check(pFunc)) {
        fprintf(stderr,
                "%s: Function \"%s\" is not callable\n", argv[0], argv[3]);
        return_value = -5;
        goto except;
    }
    
    
    fprintf(stdout, "Calling function %s\n", argv[3]);
    pResult = PyObject_CallObject(pFunc, NULL);
    if (! pResult) {
        fprintf(stderr, "%s: Function call failed\n", argv[0]);
        return_value = -6;
        goto except;
    }
#ifdef DEBUG
    printf("%s: PyObject_CallObject() succeeded\n", argv[0]);
#endif
    assert(! PyErr_Occurred());
    goto finally;
except:
    assert(PyErr_Occurred());
    PyErr_Print();
finally:
    Py_XDECREF(pFunc);
    Py_XDECREF(pModule);
    Py_XDECREF(pResult);
    Py_Finalize();
    return return_value;
}

#ifdef __cplusplus
}
#endif
