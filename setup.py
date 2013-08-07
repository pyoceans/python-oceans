#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

from ocean import __version__ as version

source = 'http://pypi.python.org/packages/source'
install_requires = ['numpy', 'scipy', 'matplotlib', 'pandas', 'gsw']

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

README = open('README.md').read()
CHANGES = open('CHANGES.txt').read()
LICENSE = open('LICENSE.txt').read()

config = dict(name='oceans',
              version=version,
              packages=['oceans', 'oceans/RPStuff', 'oceans/colormaps',
                        'oceans/datasets', 'oceans/ff_tools',
                        'oceans/plotting', 'oceans/test'],
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
              download_url='%s/s/oceans/oceans-%s.tar.gz' (source, version),
              platforms='any',
              keywords=['oceanography', 'data analysis'],
              install_requires=install_requires)

setup(**config)
