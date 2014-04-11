#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from setuptools import setup

import re
VERSIONFILE="oceans/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


source = 'http://pypi.python.org/packages/source'
install_requires = ['numpy', 'scipy', 'matplotlib', 'Shapely',
                    'netCDF4', 'pandas', 'gsw', 'seawater']

classifiers = """\
Development Status :: 2 - Pre-Alpha
Environment :: Console
Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: Education
License :: OSI Approved :: MIT License
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Education
Topic :: Software Development :: Libraries :: Python Modules
"""

README = open('README.txt').read()
CHANGES = open('CHANGES.txt').read()
LICENSE = open('LICENSE.txt').read()

config = dict(name='oceans',
              version=verstr,
              packages=['oceans', 'oceans/RPSstuff', 'oceans/colormaps',
                        'oceans/datasets', 'oceans/ff_tools',
                        'oceans/plotting', 'oceans/sw_extras', 'oceans/tests'],
              test_suite='test',
              use_2to3=True,
              package_data={'': ['colormaps/cmap_data/*.pkl']},
              license=LICENSE,
              long_description='%s\n\n%s' % (README, CHANGES),
              classifiers=filter(None, classifiers.split("\n")),
              description='Misc functions for oceanographic data analysis',
              author='Filipe Fernandes',
              author_email='ocefpaf@gmail.com',
              maintainer='Filipe Fernandes',
              maintainer_email='ocefpaf@gmail.com',
              url='http://pypi.python.org/pypi/oceans/',
              download_url='%s/s/oceans/oceans-%s.tar.gz' % (source, verstr),
              platforms='any',
              keywords=['oceanography', 'data analysis'],
              install_requires=install_requires)

setup(**config)
