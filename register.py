# -*- coding: utf-8 -*-
#
# register.py
#
# purpose:  Automagically creates a Rst README.txt
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.github.io/
# created:  10-Apr-2014
# modified: Fri 11 Apr 2014 12:10:43 AM BRT
#
# obs: https://coderwall.com/p/qawuyq
#

import os
import pandoc

home = os.path.expanduser("~")
pandoc.core.PANDOC_PATH = os.path.join(home, 'bin', 'pandoc')

doc = pandoc.Document()
doc.markdown = open('README.md').read()
with open('README.txt', 'w+') as f:
    f.write(doc.rst)

# Some modifications are need to README.txt before registering.  Rendering this
# part useless...
if False:
    os.system("python2 setup.py register")
    os.remove('README.txt')
