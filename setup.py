#!/usr/bin/env python3
""" NeuroCorgi-SDK
This SDK is used to manipulate the NeuroCorgi model in your own applications.
"""

DOCLINES = (__doc__ or '').split("\n")

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3 :: Only
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Software Development
Topic :: Software Development :: Libraries
Topic :: Software Development :: Libraries :: Python Modules
Typing :: Typed
Operating System :: POSIX
"""

import sys
from setuptools import setup
from setuptools import find_packages


# Python supported version checks
if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")


if __name__ == '__main__':

    setup(
        name='neurocorgi_sdk',
        version="1.0.0",
        url='https://github.com/CEA-LIST/neurocorgi_sdk',
        license='CECILL-2.1',
        author='CEA-List',
        python_requires='>=3.7',
        description=DOCLINES[0],
        long_description_content_type="text/markdown",
        long_description="\n".join(DOCLINES[2:]),
        keywords=['neurocorgi', 'sdk', 'machine', 'learning'],
        classifiers=[c for c in CLASSIFIERS.split('\n') if c],
        platforms=["Linux"],
        packages=find_packages(
            where=".",
        ),
    )
