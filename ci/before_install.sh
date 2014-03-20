# http://stackoverflow.com/questions/17224389/
# scipy-importerror-on-travis-ci

# Install Dependencies.
SITE_PKG_DIR=$VIRTUAL_ENV/lib/python$TRAVIS_PYTHON_VERSION/site-packages
echo "Using SITE_PKG_DIR: $SITE_PKG_DIR"

# Workaround for travis ignoring system_site_packages in travis.yml
rm -f $VIRTUAL_ENV/lib/python$TRAVIS_PYTHON_VERSION/no-global-site-packages.txt

sudo apt-get update -qq
sudo apt-get install -qq build-essential
sudo apt-get install -qq build-dep python-scipy
sudo apt-get install -qq build-dep python-numpy
sudo apt-get install python-matplotlib

sudo apt-get install -qq cython
sudo apt-get install -qq python-pip
sudo apt-get install libudunits2-dev libhdf5-serial-dev libnetcdf-dev netcdf-bin
sudo /usr/bin/pip install netCDF4

# python 3
# sudo apt-get install -qq python3-numpy
# sudo apt-get install -qq python3-scipy
