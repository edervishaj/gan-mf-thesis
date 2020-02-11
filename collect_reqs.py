#!/Users/owner/miniconda3/envs/ml/bin/python
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import os
import pip
import glob
import subprocess

if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    default_packages = set(['os', 'sys', 'itertools', 'operator', 'zipfile', 'pickle', 'subprocess', 'json', 'random', 'array'])
    not_allowed_packages = set(['datasets', 'Scripts', 'Utils_'])
    packages = []
    with open('requirements.txt', 'w') as f:
        # Go over all files ending in *.py and collect only trimmed lines starting with `import` and `from`
        for fname in glob.iglob(cwd + '/**/*.py', recursive=True):
            # Disregard this file
            if fname == os.path.abspath(__file__):
                continue
            with open(fname, 'r') as g:
                # Read all lines of every files ending in *.py
                for l in g.readlines():
                    tokenized_line = l.strip().split(' ')
                    if len(tokenized_line) > 0:
                        if tokenized_line[0] in ['import', 'from']:
                            complex_package = tokenized_line[1]
                            # Append the package only
                            packages.append(complex_package.split('.')[0])
        packages = list(set(packages).difference(not_allowed_packages).difference(default_packages))
        grep_str = (''.join([x + '\|' for x in packages]))[:-2] + '\"'
        p1 = subprocess.Popen(['pip', 'freeze'], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(['grep', grep_str], stdin=p1.stdout, stdout=subprocess.PIPE, encoding='ASCII')
        p1.stdout.close()
        output = p2.communicate()[0]
        f.writelines([line + '\n' for line in output.split()])