import os

from setuptools import find_packages, setup

import versioneer

rootpath = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    return open(os.path.join(rootpath, *parts), 'r').read()


email = 'ocefpaf@gmail.com'
maintainer = 'Filipe Fernandes'
authors = [u'André Palóczy', 'Arnaldo Russo', 'Filipe Fernandes']

LICENSE = read('LICENSE.txt')
long_description = '{}\n{}'.format(read('README.rst'), read('CHANGES.txt'))

# Dependencies.
hard = ['gsw', 'matplotlib', 'numpy', 'seawater']
soft = {'full': ['cartopy' 'iris', 'netcdf4', 'pandas', 'scipy']}

setup(
    name='oceans',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    package_data={'oceans': ['colormaps/cmap_data/*.dat']},
    license=LICENSE,
    long_description=long_description,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        ],
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
    tests_require=['pytest'],
)
