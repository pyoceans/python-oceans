#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

rootpath = os.path.abspath(os.path.dirname(__file__))


class PyTest(TestCommand):
    """python setup.py test"""
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--strict', '--verbose', '--tb=long', 'tests']
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


def read(*parts):
    return open(os.path.join(rootpath, *parts), 'r').read()


def extract_version():
    version = None
    fname = os.path.join(rootpath, 'oceans', '__init__.py')
    with open(fname) as f:
        for line in f:
            if (line.startswith('__version__')):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Remove quotation characters
                break
    return version

email = "ocefpaf@gmail.com"
maintainer = "Filipe Fernandes"
authors = [u'André Palóczy', 'Arnaldo Russo', 'Filipe Fernandes']

LICENSE = read('LICENSE.txt')
long_description = '{}\n{}'.format(read('README.txt'), read('CHANGES.txt'))

# Dependencies.
hard = ['numpy', 'matplotlib', 'gsw', 'seawater']
soft = dict(full=["scipy", "Shapely", "netCDF4", "pandas"])
tests_require = ['pytest', 'pytest-cov']

config = dict(name='oceans',
              version=extract_version(),
              packages=['oceans', 'oceans/RPSstuff', 'oceans/colormaps',
                        'oceans/datasets', 'oceans/ff_tools',
                        'oceans/plotting', 'oceans/sw_extras', 'oceans/tests'],
              package_data={'': ['colormaps/cmap_data/*.dat']},
              cmdclass=dict(test=PyTest),
              license=LICENSE,
              long_description=long_description,
              classifiers=['Development Status :: 4 - Beta',
                           'Environment :: Console',
                           'Intended Audience :: Science/Research',
                           'Intended Audience :: Developers',
                           'Intended Audience :: Education',
                           'License :: OSI Approved :: MIT License',
                           'Operating System :: OS Independent',
                           'Programming Language :: Python',
                           'Topic :: Education',
                           'Topic :: Scientific/Engineering'],
              description='Misc functions for oceanographic data analysis',
              author=authors,
              author_email=email,
              maintainer='Filipe Fernandes',
              maintainer_email=email,
              url='https://pypi.python.org/pypi/oceans/',
              platforms='any',
              keywords=['oceanography', 'data analysis'],
              extras_require=soft,
              install_requires=hard,
              tests_require=tests_require,
              use_2to3=True)  # FIXME: Remove 2to3.

setup(**config)
