#!/usr/bin/env python
# -*- coding: utf-8 -*-

from oceans import __version__

from distutils.core import setup

try:  # Python 3
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2
    from distutils.command.build_py import build_py


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

config = dict(name='oceans',
              version=__version__,
              packages=[
                        'oceans',
                        'oceans/RPStuff',
                        'oceans/colormaps',
                        'oceans/ctd',
                        'oceans/datasets',
                        'oceans/ff_tools',
                        'oceans/mlabwrap',
                        'oceans/plotting',
                        'oceans/test'],
              package_data={'': ['colormaps/cmap_data/*.pkl']},
              license=open('LICENSE.txt').read(),
              description='Module for oceanographic data analysis',
              long_description=open('README.rst').read(),
              author='Filipe Fernandes',
              author_email='ocefpaf@gmail.com',
              maintainer='Filipe Fernandes',
              maintainer_email='ocefpaf@gmail.com',
              url='http://pypi.python.org/pypi/oceans/',
              download_url='http://pypi.python.org/packages/source/s/oceans/',
              classifiers=filter(None, classifiers.split("\n")),
              platforms='any',
              cmdclass={'build_py': build_py},
              keywords=['oceanography', 'data analysis'],
              install_requires=['numpy', 'scipy', 'nose'])

setup(**config)
