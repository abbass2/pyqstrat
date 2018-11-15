from __future__ import print_function
from setuptools import setup, find_packages, Extension
from setuptools.command.test import test as TestCommand
from setuptools.command.build_ext import build_ext
import setuptools
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

if 'CONDA_PREFIX' in os.environ:

    if sys.platform in ["win32", "cygwin"]:
        include_dirs += [os.environ['CONDA_PREFIX'] + '\\include',
                         os.environ['CONDA_PREFIX'] + '\\Library\\include']
    else:
        include_dirs.append(os.environ['CONDA_PREFIX'] + '/include')

    if sys.platform in ["win32", "cygwin"]:
        library_dirs += [os.environ['CONDA_PREFIX'] + '\\lib',
                         os.environ['CONDA_PREFIX'] + '\\Library\\lib',
                         os.environ['CONDA_PREFIX'] + '\\bin',
                         os.environ['CONDA_PREFIX'] + '\\Library\\bin']
        
    else:
        library_dirs = [os.environ['CONDA_PREFIX'] + '/lib']

    extra_link_args = None
    if sys.platform == 'darwin':
        link_dirs = ',-rpath,'.join(library_dirs)
        extra_link_args=[f'-Wl,-rpath,{link_dirs}']

libraries = [
    'z',
    'arrow',
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
            'pyqstrat/cpp/arrow_writer.cpp',
            'pyqstrat/cpp/tests.cpp',
            'pyqstrat/cpp/text_file_processor.cpp',
            'pyqstrat/cpp/pybind.cpp',
            'pyqstrat/cpp/py_import_call_execute.cpp'
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
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-Ofast'):
                opts.append('-Ofast')
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
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

setup(
    name='pyqstrat',
    version=__version__,
    author='sal',
    author_email='abbasi.sal@gmail.com',
    url='http://github.com/abbass2/pyqstrat/',
    license='BSD',
    tests_require=['pytest'],
    python_requires='>=3.6',
    install_requires=['pandas>=0.22',
                      'numpy>=1.14',
                      'matplotlib>=2.2.2',
                      'scipy >= 1.0.0',
		      'ipython>=6.5.0',
                      'pybind11>=2.2',
                      'pyarrow>=0.1.0'
                    ],
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: Only',
        ],
    extras_require={
        'testing': ['pytest'],
    },
    cmdclass = {'test': PyTest, 'build_ext' : BuildExt},
    zip_safe = False
)
