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

## @author Staffan Ronnas

import os
import re
import string
import sys

#-------------------------------------------------------------------------------
# no email-adresses in source code

#    check only outside of ./contrib
#
#    which files count as source code?  currently *.h *.cc *.py *.pl *.sh AUTHORS

#-------------------------------------------------------------------------------
# do all source code files have a valid license?

license_check_exceptions = [
    "./utils/cpplint.py",
    "./utils/gprof2dot.py",
    "./src/linear_algebra/lmp/mmio.h"]

oldlicstr = r"""// Copyright (C) 2011-2015 Vincent Heuveline
//
// HiFlow3 is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//
// HiFlow3 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with HiFlow3.  If not, see <http://www.gnu.org/licenses/>.
"""

licstr = r"""// Copyright (C) 2011-2021 Vincent Heuveline
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
// along with HiFlow3.  If not, see <https://joinup.ec.europa.eu/page/eupl-text-11-12>.
"""

# Change this string to search for different regexp.
rawstr = licstr

license_match_obj = re.compile(rawstr,  re.MULTILINE)

def find_files(search_missing):
    result = []
    for dirpath, dirnames, filenames in os.walk('.', topdown=True):
        if dirpath.startswith("./contrib/") or dirpath.startswith('./.git') or dirpath.startswith('./doc'):
            dirnames = []
        else:
            for name in filenames:
                path = dirpath + '/' + name
                if path in license_check_exceptions:
                    continue

                if re.match(r'.*\.(h|c|cpp|cu|cc|h.in|dox)$', name):
                    # print dirpath + '/' + name
                    found_match = license_match_obj.search(open(path, "r").read(2000))
                    if found_match and not search_missing:
                        result.append(path)
                    if not found_match and search_missing:
                        result.append(path)

    return result

def add_license(path):
     with open(path, "r") as f:
        s = f.read()
     s = licstr + "\n" + s

     with open(path, "w") as f:
        f.write(s)

def remove_license(path):
    with open(path, "r") as f:
        s = f.read()

    s = string.replace(s, oldlicstr, "")

    with open(path, "w") as f:
        f.write(s)

if __name__ == '__main__':

    including_paths = find_files(False)
    missing_paths = find_files(True)

    if len(sys.argv) < 2:

        print "================================================="
        print "The following files contain the old license text:"
        for p in including_paths:
            print p

        # print "================================================="
        # print "The following files do not contain the old license text:"
        # for p in missing_paths:
        #     print p
        # print "================================================="

        print "Number of files with old license text: %d " % (len(including_paths))
        print "Number of files with new license text: %d " % (len(missing_paths))

    if len(sys.argv) == 2:
        if sys.argv[1] == "remove":
            for p in missing_paths:
                remove_license(p)
        elif sys.argv[1] == "add":
            for p in missing_paths:
                add_license(p)
        else:
            print "Usage: %s (remove|add)" % (sys.argv[0])
