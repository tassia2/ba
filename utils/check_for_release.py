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
import re

#-------------------------------------------------------------------------------
# no email-adresses in source code

#    check only outside of ./contrib
#
#    which files count as source code?  currently *.h *.cc *.py *.pl *.sh AUTHORS

# find -not -path './contrib*' -name \*.h -or -name \*.cc -print0 | xargs -0 egrep '[[:alnum:]]+@[[:alnum:].]+'

p1 = Popen("find -not -path ./contrib* ( -name *.h -or -name *.cc -or -name *.py -or -name *.pl -or -name *.sh -name AUTHORS ) -print0".split(' '), stdout=PIPE)
p2 = Popen("xargs -0 egrep [[:alnum:]]+@[[:alnum:].]+".split(' '), stdin=p1.stdout, stdout=PIPE)

result = p2.communicate()[0]
if len(result) > 0:
    print
    print "==============================================================================="
    print "still e-mail adresses in files:"
    print
    print result

#-------------------------------------------------------------------------------
# do headerfiles have an author?

# find -not -path './contrib*' -name \*.h -print0 | xargs -0 egrep -L '///.*[\\@]author'

p1 = Popen("find -not -path ./contrib* -not -path ./doc/tutorials/elasticity/Code/* -name *.h -print0".split(' '), stdout=PIPE)
p2 = Popen("xargs -0 egrep -L ///.*[\@]author".split(' '), stdin=p1.stdout, stdout=PIPE)

result = p2.communicate()[0]
if len(result) > 0:
    print
    print "==============================================================================="
    print "still header files without author:"
    print
    print result

#-------------------------------------------------------------------------------
# do all source code files have a valid license?

license_check_exceptions = [
    "./utils/cpplint.py",
    "./utils/gprof2dot.py",
    "./src/linear_algebra/lmp/mmio.h"]

rawstr = r"""// Copyright (C) 2011-2021 Vincent Heuveline
//
// HiFlow3 is free software: you can redistribute it and/or modify it under the
// terms of the European Union Public Licence (EUPL) v1.2 as published by the
// European Union or (at your option) any later version.
//
// HiFlow3 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the European Union Public Licence (EUPL) v1.2 for more
// details.
//
// You should have received a copy of the European Union Public Licence (EUPL) v1.2
// along with HiFlow3.  If not, see <https://joinup.ec.europa.eu/page/eupl-text-11-12>."""

license_match_obj = re.compile(rawstr,  re.MULTILINE)
result = []

for dirpath, dirnames, filenames in os.walk('.', topdown=True):
    if dirpath.startswith("./contrib/") or dirpath.startswith('./.git') or dirpath.startswith('./doc/tutorials/elasticity/Code'):
        dirnames = []
    else:
        for name in filenames:
            if re.match(r'.*\.(h|c|py|cpp|pl)$', name):
                #print dirpath + '/' + name
                if not license_match_obj.search(open(dirpath + '/' + name, "r").read(1000)):
                    cand = dirpath + '/' + name
                    if not cand in license_check_exceptions:
                        result.append(dirpath + '/' + name)

if len(result) > 0:
    print
    print "==============================================================================="
    print "still files with missing or old license:"
    print
    for name in result:
        print name
