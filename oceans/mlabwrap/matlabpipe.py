#!/usr/bin/env python

r"""A python module for raw communication with Matlab(TM) using pipes under
unix.  This module exposes the same interface as mlabraw.cpp, so it can be used
along with mlabwrap.py.  The module sends commands to the matlab process using
the standard input pipe.  It loads data from/to the matlab process using the
undocumented save/load stdio commands. Only unix (or mac osx) versions of
Matlab support pipe communication, so this module will only work under unix
(or mac osx).

Author: Dani Valevski <daniva@gmail.com>
Dependencies: scipy
Tested Matlab Versions: 2009b, 2010a, 2010b, 2011a
License: MIT"""

import os
import sys
import fcntl
import select
import subprocess

from cStringIO import StringIO

import numpy as np
import scipy.io as mlabio


class MatlabError(Exception):
    r"""Raised when a Matlab evaluation results in an error inside Matlab."""
    pass


class MatlabConnectionError(Exception):
    r"""Raised for errors related to the Matlab connection."""
    pass


def find_matlab_process(binary='matlab'):
    r""""Tries to guess matlab process path."""
    cmd = ['which', binary]
    try:
        return subprocess.check_output(cmd).strip()
    except:
        return None


def find_matlab_version(matlab_version):
    r"""Tries to guess matlab's version according to its process path.
    If we couldn't guess the version, None is returned."""
    #version = $(matlab -nodesktop -nodisplay -nosplash -nojvm -r \
            #"try, version -release ;quit")
    version = None
    if not is_valid_version_code(matlab_version):
        return None
    return version


def is_valid_version_code(version):
    r"""Checks that the given version code is valid."""
    return (version is not None and len(version) == 5 and
            int(version[:4]) in range(1990, 2050) and
            version[4] in ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'])


class MatlabPipe(object):
    r"""Manages a connection to a matlab process.

    Matlab version is in the format [YEAR][VERSION] for example: 2011a.
    The process can be opened and close with the open/close methods.
    To send a command to the matlab shell use 'eval'.
    To load numpy data to the matlab shell use 'put'.
    To retrieve numpy data from the matlab shell use 'get'."""

    def __init__(self, matlab_process_path=None, matlab_version=None):
        if matlab_process_path and matlab_version:
            if matlab_process_path == 'guess':
                matlab_process_path = find_matlab_process()
            if matlab_version == 'guess':
                matlab_version = find_matlab_version(matlab_process_path)
            if not is_valid_version_code(matlab_version):
                raise ValueError('Invalid version code %s' % matlab_version)
            if not os.path.exists(matlab_process_path):
                raise ValueError('Matlab process path %s does not exist' %
                                 matlab_process_path)
        else:
            raise NameError('''Matlab process path and/or version must be
                               explicitly specified or use "guess".''')
        self.matlab_version = (int(matlab_version[:4]), matlab_version[4])
        self.matlab_process_path = matlab_process_path
        self.process = None
        self.command_end_string = '___MATLAB_PIPE_COMMAND_ENDED___'
        self.expected_output_end = '%s\n>> ' % self.command_end_string
        self.stdout_to_read = ''

    def open(self, print_matlab_welcome=False):
        r"""Opens the matlab process."""
        # TODO: -nojvm, -nodesktop, -nosplash, and -nodisplay as options
        if self.process and not self.process.returncode:
            raise MatlabConnectionError('Matlab(TM) process is still active. '
                                        'Use close to close it')
        self.process = subprocess.Popen([self.matlab_process_path,
                                         '-nojvm',
                                         '-nodesktop',
                                         '-nosplash',
                                         '-nodisplay'],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
        flags = fcntl.fcntl(self.process.stdout, fcntl.F_GETFL)
        fcntl.fcntl(self.process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        if print_matlab_welcome:
            self._sync_output()
        else:
            self._sync_output(None)

    def close(self):
        r"""Closes the matlab process."""
        self._check_open()
        self.process.stdin.close()

    def eval(self, expression, identify_errors=True, print_expression=True,
             on_new_output=sys.stdout.write):
        r"""Evaluates a matlab expression synchronously.

        If identify_erros is true, and the last output line after evaluating
        the expressions begins with '???' and exception is thrown with the
        matlab error following the '???'.  If on_new_output is not None, it
        will be called whenever a new output is encountered.  The default value
        prints the new output to the screen.  The return value of the function
        is the matlab output following the call."""

        self._check_open()
        if print_expression:
            print expression

        self.process.stdin.write(expression)
        self.process.stdin.write('\n')
        ret = self._sync_output(on_new_output)
        # TODO: Use stderr to identify errors.
        if identify_errors and ret.rfind('???') != -1:
            begin = ret.rfind('???') + 4
            end = ret.find('\n', begin)
            raise MatlabError(ret[begin:end])
        return ret

    def put(self, name_to_val, oned_as='row', on_new_output=None):
        r"""Loads a dictionary of variable names into the matlab shell.

        oned_as is the same as in scipy.io.matlab.savemat function:
        oned_as : {'column', 'row'}, optional
        If 'column', write 1-D numpy arrays as column vectors.
        If 'row', write 1D numpy arrays as row vectors."""
        self._check_open()
        # We can't give stdin to mlabio.savemat because it needs random access.
        temp = StringIO()
        mlabio.savemat(temp, name_to_val, oned_as=oned_as)
        temp.seek(0)
        temp_str = temp.read()
        temp.close()
        self.process.stdin.write('load stdio;\n')
        self._read_until('ack load stdio\n', on_new_output=on_new_output)
        self.process.stdin.write(temp_str)
        #print 'sent %d kb' % (len(temp_str) / 1024)
        self._read_until('ack load finished\n', on_new_output=on_new_output)
        self._sync_output(on_new_output=on_new_output)

    def get(self, names_to_get, extract_numpy_scalars=True,
            on_new_output=None):
        r"""Loads the requested variables from the matlab shell.

        names_to_get can be either a variable, a list of variable names, or
        None.  If it is a variable name, the values is returned.  If it is a
        list, a dictionary of variable_name -> value is returned.  If it is
        None, a dictionary with all variables is returned.  If
        extract_numpy_scalars is true, the method will convert numpy scalars
        (0-dimension arrays) to a regular python variable."""

        self._check_open()
        single_itme = isinstance(names_to_get, (unicode, str))
        if single_itme:
            names_to_get = [names_to_get]
        if names_to_get:
            self.process.stdin.write('save stdio;\n')
        else:
            # Make sure that we throw an exception if names are not defined.
            for name in names_to_get:
                self.eval('%s;' % name, print_expression=False,
                          on_new_output=on_new_output)
        self.process.stdin.write("save('stdio', '%s', '-v6');\n" %
                                 '\', \''.join(names_to_get))
        # We have to read to a temp buffer because mlabio.loadmat needs
        # random access :(
        self._read_until('start_binary\n', on_new_output=on_new_output)
        temp_str = self._sync_output(on_new_output=on_new_output)
        # Remove expected output and "\n>>"
        # TODO: Get rid of the unnecessary copy.
        # MATLAB 2010a adds an extra >> so we need to remove more spaces.
        if self.matlab_version == (2010, 'a'):
            temp_str = temp_str[:-len(self.expected_output_end) - 6]
        else:
            temp_str = temp_str[:-len(self.expected_output_end) - 3]
        temp = StringIO(temp_str)
        ret = mlabio.loadmat(temp, chars_as_strings=True, squeeze_me=True)
        temp.close()
        for key in ret.iterkeys():
            while ret[key].shape and ret[key].shape[-1] == 1:
                ret[key] = ret[key][0]
        if extract_numpy_scalars:
            if isinstance(ret[key], np.ndarray) and not ret[key].shape:
                ret[key] = ret[key].tolist()
        if single_itme:
            return ret.values()[0]
        return ret

    def _check_open(self):
        if not self.process or self.process.returncode:
            raise MatlabConnectionError('Matlab(TM) process is not active.')

    def _read_until(self, wait_for_str, on_new_output=sys.stdout.write):
        all_output = StringIO()
        output_tail = self.stdout_to_read
        while not wait_for_str in output_tail:
            tail_to_remove = output_tail[:-len(output_tail)]
            output_tail = output_tail[-len(output_tail):]
            if on_new_output:
                on_new_output(tail_to_remove)
            all_output.write(tail_to_remove)
            if not select.select([self.process.stdout], [], [], 20)[0]:
                raise MatlabConnectionError('timeout')
            new_output = self.process.stdout.read(65536)
            output_tail += new_output
        chunk_to_take, chunk_to_keep = output_tail.split(wait_for_str, 1)
        chunk_to_take += wait_for_str
        self.stdout_to_read = chunk_to_keep
        if on_new_output:
            on_new_output(chunk_to_take)
        all_output.write(chunk_to_take)
        all_output.seek(0)
        return all_output.read()

    def _sync_output(self, on_new_output=sys.stdout.write):
        self.process.stdin.write('disp(\'%s\');\n' % self.command_end_string)
        return self._read_until(self.expected_output_end, on_new_output)

if __name__ == '__main__':
    import unittest

    class TestMatlabPipe(unittest.TestCase):
        def setUp(self):
            self.matlab = MatlabPipe(matlab_process_path='guess',
                                     matlab_version='2010b')

            self.matlab = MatlabPipe(matlab_process_path='guess',
                                     matlab_version='guess')  # Not working.
            self.matlab = MatlabPipe(matlab_process_path='guess')
            self.matlab.open()

        def tearDown(self):
            self.matlab.close()

        def test_eval(self):
            for i in xrange(100):
                ret = self.matlab.eval('disp \'hiush world%s\';' % ('b' * i))
                self.assertTrue('hiush world' in ret)

        def test_put(self):
            self.matlab.put({'A': [1, 2, 3]})
            ret = self.matlab.eval('A')
            self.assertTrue('A =' in ret)

        def test_1_element(self):
            self.matlab.put({'X': 'string'})
            ret = self.matlab.get('X')
            self.assertEquals(ret, 'string')

        def test_get(self):
            self.matlab.eval('A = [1 2 3];')
            ret = self.matlab.get('A')
            self.assertEquals(ret[0], 1)
            self.assertEquals(ret[1], 2)
            self.assertEquals(ret[2], 3)

        def test_error(self):
            self.assertRaises(MatlabError, self.matlab.eval,
                              'no_such_function')

    unittest.main()
