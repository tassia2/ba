#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2011-2021 Vincent Heuveline
#
# HiFlow3 is free software: you can redistribute it and/or modify it under the
# terms of the European Union Public Licence (EUPL) v1.2 as published by the
#/ European Union or (at your option) any later version.
#
# HiFlow3 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the European Union Public Licence (EUPL) v1.2 for more
# details.
#
# You should have received a copy of the European Union Public Licence (EUPL) v1.2
# along with HiFlow3.  If not, see <https://joinup.ec.europa.eu/page/eupl-text-11-12>.

## @author Werner Augustin

import os
from subprocess import *
import string
import re
import sys

if len(sys.argv) != 3 :
    print "usage: make_release.py [srcdir] [reldir]"
    print "example: make_release.py /home/itsme/hiflow3 /home/itsme/hiflow-1.1"
    sys.exit(1)

srcdir = os.path.abspath(sys.argv[1])
reldir = sys.argv[2].rstrip('/')

print "srcdir:", srcdir
print "reldir:", reldir

## os.system("rm -rf %s" % reldir)

print "rsync -aP --delete --exclude=.git %s/. %s" % (srcdir, reldir)

os.system("rsync -aP --delete --exclude=.git %s/. %s" % (srcdir, reldir))

# remove git-files

os.chdir(reldir)
print os.getcwd()
os.system("rm -rf ./.git")
os.system("find -name '.gitignore' -print0 | xargs -0 rm -rf")

# remove application folder
print os.getcwd()
os.system("rm -rf ./application")

# remove possible build directory
print os.getcwd()
os.system("rm -rf ./build")

run_pdflatex = True

#-------------------------------------------------------------------------------
# make tutorials
cwd = os.getcwd()
os.chdir("%s/utils/" % srcdir)
os.system("python make_tutorial.py " + reldir)
os.chdir(cwd)

#-------------------------------------------------------------------------------
# check test folder
'''

#- Check ob Tutorial vorhanden -> falls nicht Meckern!

testdata = open("test/readme.txt", "r").read()
files_needed_for_test = set()
potentially_delete_files = set()
tests_to_delete = set()

parse_test_readme = re.compile(\
r"""^Test:\s*(.*?)
^Author:.*?
^Needed Data:(.*?)
^OpenSource:(.*?)
(?=^Test:|\Z)""",  re.MULTILINE | re.DOTALL)

for m in parse_test_readme.finditer(testdata):
    if m.group(3).strip().lower() in 'yes': # OpenSource
        files_needed_for_test = files_needed_for_test.union(m.group(2).split())

    else:
        tests_to_delete = tests_to_delete.union(m.group(1).split())
        potentially_delete_files = potentially_delete_files.union(m.group(2).split())

files_needed_for_test.discard('')
tests_to_delete.discard('')
potentially_delete_files.discard('')

print "==============================================================================="

print "deleting tests because of closed source license:"
print ' '.join(tests_to_delete)
for n in tests_to_delete:
    filename = 'test/' + n
    if os.path.exists(filename): os.remove(filename)
    else:
        print "warning: %s mentioned in test/readme.txt doesn't exist" % (filename,)
print

for n in files_needed_for_test:
    filename = 'test/data/' + n
    if not os.path.exists(filename):
        print filename, 'is missing but referenced in test/readme.txt'

#===============================================================================
# check examples folder

examples_data = open("examples/readme.txt", "r").read()
files_needed_for_examples = set()
examples_to_delete = set()

parse_examples_readme = re.compile(\
r"""^Example:\s*(.*?)
^Author:.*?
^Needed Data:(.*?)
^OpenSource:(.*?)
^Documentation:(.*?)
(?=^Example:|\Z)""",  re.MULTILINE | re.DOTALL)

for m in parse_examples_readme.finditer(examples_data):
    names_of_examples = m.group(1).split()
    if m.group(3).strip().lower() in 'yes': # OpenSource
        files_needed_for_examples = files_needed_for_examples.union(m.group(2).split())

        documentation = m.group(4).strip()
        if documentation != 'no' and documentation != 'yes': # run pdflatex for documentation
            documentation_list = documentation.split()
            for name in documentation_list:
                cwd = os.getcwd()
                os.chdir(os.path.dirname(name))
                bname = os.path.basename(name)
                print "pdflatex %s" % bname
                if( run_pdflatex ):
                    assert( os.system("pdflatex %s" % bname                      ) == 0)
                    assert( os.system("bibtex %s" % bname.rstrip('.tex')         ) == 0)
                    assert( os.system("makeindex %s.idx" % bname.rstrip('.tex' ) ) == 0)
                    assert( os.system("pdflatex %s" % bname                      ) == 0)
                    assert( os.system("pdflatex %s" % bname                      ) == 0)
                os.chdir(cwd)
                if( run_pdflatex ):
                    command =  "cp %s %s" % (name.rstrip('.tex')+'.pdf',
                                             'examples/' + os.path.dirname(names_of_examples[0]) + '/' +
                                             os.path.basename(name).lstrip('tut_').rstrip('.tex')+'.pdf')
                    print command ; assert( os.system(command) == 0 )

    else:
        examples_to_delete = examples_to_delete.union(names_of_examples)
        potentially_delete_files = potentially_delete_files.union(m.group(2).split())

files_needed_for_examples.discard('')
examples_to_delete.discard('')
potentially_delete_files.discard('')

print
print "==============================================================================="

print "deleting examples because of closed source license:"
print
print ' '.join(examples_to_delete)
for n in examples_to_delete:
    filename = 'examples/' + n
    if os.path.exists(filename): os.remove(filename)
    else: print "warning: %s mentioned in examples/readme.txt doesn't exist" % (filename,)
print

# files_to_delete = potentially_delete_files.difference(files_needed_for_test).difference(files_needed_for_examples)
#
# print "deleting unneeded data files:"
# print ' '.join(files_to_delete)
# for n in files_to_delete:
#     filename = 'test/data/' + n
#     if os.path.exists(filename): os.remove(filename)
#
# print

cwd = os.getcwd()
os.chdir("test/data")
for dirpath, dirnames, filenames in os.walk('.', topdown=True):
    for name in filenames:
        if (not (name in files_needed_for_test) and
            not (name in files_needed_for_examples)   ):
            print "removing", name
            if os.path.exists(name): os.remove(name)
os.chdir(cwd)

for n in files_needed_for_examples:
    filename = 'test/data/' + n
    if not os.path.exists(filename):
        print filename, 'is missing but referenced int examples/readme.txt'

'''
#===============================================================================
# delete source files of documentation

print "==============================================================================="
print "delete source files of documentation"
print

#command = "rm -rf doc/tutorials";             print command; os.system(command)
#command = "rm -rf doc/POOSC10";               print command; os.system(command)
command = "rm -rf doc/doxygen/html";          print command; os.system(command)

#===============================================================================
# delete parts of CMakelists.txt
# old stuff, should not do anything
'''
print "==============================================================================="
print "delete parts of CMakelists.txt"
print

remove_pattern =\
"""
^# REMOVE BEFORE RELEASE BEGIN.*?
# REMOVE BEFORE RELEASE END$"""
# old stuff to remove things that are not for publication
# should not do anything any more...

remove_re_obj = re.compile(remove_pattern, re.MULTILINE | re.DOTALL )

for dirpath, dirnames, filenames in os.walk('.', topdown=True):
    for name in filenames:
        if name ==  "CMakeLists.txt":
            matchstr = open(dirpath + '/' + name, "r").read()
            newstr, nrepl = remove_re_obj.subn("", matchstr, 0)
            if nrepl > 0:
                print "delete parts in " + dirpath + '/' + name
                open(dirpath + '/' + name, "w").write(newstr)

'''
#===============================================================================
# delete misc files

#os.remove("utils/gprof2dot.py")
os.remove("utils/check_for_release.py")
os.remove("utils/make_release.py")
os.remove("utils/make_tutorial.py")
os.remove("utils/update_license_header.py")

#===============================================================================
# create tar

print
print "==============================================================================="
print "creating archives"
print

cwd = os.getcwd()
os.chdir("..")
command = "zip -r %s.zip %s" % (2*(os.path.basename(reldir),))
print command; os.system(command)
command = "tar czf %s.tar.gz %s" % (2*(os.path.basename(reldir),))
print command; os.system(command)
command = "tar cjf %s.tar.bz2 %s" % (2*(os.path.basename(reldir),))
print command; os.system(command)
# command = "tar cJf %s.tar.xz %s" % (2*(os.path.basename(reldir),))
# print command; os.system(command)
