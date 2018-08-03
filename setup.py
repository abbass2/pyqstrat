from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import codecs
import os
import sys
import pyqstrat

if sys.version_info < (3, 6, 0):
    raise RuntimeError("pyqstrat requires Python 3.6 or higher")

here = os.path.abspath(os.path.dirname(__file__))

__version__ = None
exec(open(os.path.join(here, 'pyqstrat', 'version.py')).read())


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.rst', 'CHANGES.txt')

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
    url='http://github.com/saabbasi/pyqstrat/',
    license='BSD',
    tests_require=['pytest'],
    python_requires='>=3.6',
    install_requires=['pandas>=0.22',
                      'numpy>=1.14',
                      'matplotlib>=2.2.2',
                      'scipy >= 1.0.0',
		      'ipython>=6.5.0'
                    ],
    description='fast / extensible library for backtesting quantitative strategies',
    long_description=long_description,
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
    cmdclass = {'test': PyTest},
)
