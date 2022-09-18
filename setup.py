from __future__ import print_function
from setuptools import setup, find_packages, Extension
from setuptools.command.test import test as TestCommand
from setuptools.command.build_ext import build_ext
import setuptools
import subprocess
import io
import codecs
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))

__version__ = None
exec(open(os.path.join(here, 'pyqstrat', 'version.py')).read())

# pybind11

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        try:
            import pybind11
        except ImportError:
            if subprocess.call([sys.executable, '-m', 'pip', 'install', 'pybind11']):
                raise RuntimeError('pybind11 install failed.')

        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

include_dirs=[
    # Path to pybind11 headers
    get_pybind_include(),
    get_pybind_include(user=True)
]

library_dirs = []
extra_link_args=[]

# Don't know what the difference is but sometimes one is set to the active environment, other times the other one is
if 'CONDA_PREFIX' in os.environ or 'CONDA_PREFIX_1' in os.environ:
    if 'CONDA_PREFIX_1' in os.environ and 'envs' in os.environ['CONDA_PREFIX_1']:
        conda_prefix = os.environ['CONDA_PREFIX_1']
    else:
        conda_prefix = os.environ['CONDA_PREFIX']

    if sys.platform in ["win32", "cygwin"]:
        include_dirs += [conda_prefix + '\\include',
                         conda_prefix + '\\Library\\include']
    else:
        include_dirs.append(conda_prefix + '/include')

    if sys.platform in ["win32", "cygwin"]:
        library_dirs += [conda_prefix + '\\lib',
                         conda_prefix + '\\Library\\lib',
                         conda_prefix + '\\bin',
                         conda_prefix + '\\Library\\bin']
        
    else:
        library_dirs = [conda_prefix + '/lib']

else:
    print(f'CONDA_PREFIX or CONDA_PREFIX_1 environment variables not found for including and linking to boost header files.')


if sys.platform in ["unix"]:
    extra_link_args.append("-D_GLIBCXX_USE_CXX11_ABI=0")

if sys.platform in ["darwin"]:
    extra_link_args.append("-stdlib=libc++")
    extra_link_args.append("-mmacosx-version-min=10.7")
    link_dirs = ',-rpath,'.join(library_dirs)
    extra_link_args.append(f'-Wl,-rpath,{link_dirs}')

libraries = [
    'zip',
    'hdf5_cpp',
    'hdf5'
]

if sys.platform not in ["win32", "cygwin"]:
    libraries.append('boost_iostreams') # Problems with linking to iostreams in windows with conda and vcpkg 

ext_modules = [
    Extension(
        'pyqstrat.pyqstrat_cpp',
        [
            'pyqstrat/cpp/utils.cpp',
            'pyqstrat/cpp/aggregators.cpp',
            'pyqstrat/cpp/text_file_parsers.cpp',
            'pyqstrat/cpp/tests.cpp',
            'pyqstrat/cpp/text_file_processor.cpp',
            'pyqstrat/cpp/pybind.cpp',
            'pyqstrat/cpp/py_import_call_execute.cpp',
            'pyqstrat/cpp/pybind_options.cpp',
            'pyqstrat/cpp/options.cpp',
            'pyqstrat/cpp/file_reader.cpp',
            'pyqstrat/cpp/zip_reader.cpp',
            'pyqstrat/cpp/hdf5_writer.cpp',
            'pyqstrat/cpp/lets_be_rational/normaldistribution.cpp',
            'pyqstrat/cpp/lets_be_rational/erf_cody.cpp',
            'pyqstrat/cpp/lets_be_rational/rationalcubic.cpp',
            'pyqstrat/cpp/lets_be_rational/lets_be_rational.cpp',
            'pyqstrat/cpp/test_quote_pair.cpp'
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries = libraries,
        language='c++',
        extra_link_args=extra_link_args
    ),
]

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    """Return the -std=c++14 compiler flag."""
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++14 support '
                           'is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct != 'msvc': opts.append("-Wno-return-std-move") # Annoying warnings from pybind11/numpy.h
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            #opts.append('-D_GLIBCXX_USE_CXX11_ABI=0') # ABI for std::string changed in C++11.  See https://stackoverflow.com/questions/34571583/understanding-gcc-5s-glibcxx-use-cxx11-abi-or-the-new-abi
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-Ofast'):
                opts.append('-Ofast')
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            opts.append('/DH5_BUILT_AS_DYNAMIC_LIB') # For windows, we need this so we link against dynamic hdf5 lib instead of static linking
        for ext in self.extensions:
            ext.extra_compile_args = opts
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes") # Bug in distutils See https://bugs.python.org/issue1222585
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)

# End pybind11

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.rst')

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pyqstrat',
    version=__version__,
    author='sal',
    author_email='abbasi.sal@gmail.com',
    url='http://github.com/abbass2/pyqstrat/',
    license='BSD',
    tests_require=['pytest'],
    python_requires='>=3.7',
    install_requires=requirements,
    description='fast / extensible library for backtesting quantitative strategies',
    long_description=long_description,
    ext_modules=ext_modules,
    packages=['pyqstrat'],
    include_package_data=True,
    platforms='any',
    test_suite='pyqstrat.test.test_pyqstrat',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'Topic :: Office/Business :: Financial :: Investment',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        ],
    extras_require={
        'testing': ['pytest'],
    },
    cmdclass = {'test': PyTest, 'build_ext' : BuildExt},
    zip_safe = False
)
