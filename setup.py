#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : setup.py
* @Author  : Wei Zhou                                     *
* @Date    : 2024/05/30 05:11:25                          *
* @E-mail  : welljoea@gmail.com                           *
* @Version : --                                           *
* You are using the program scripted by Wei Zhou.         *
* Please give me feedback if you find any problems.       *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''

import os
import re
import subprocess
from setuptools import find_packages,  setup, Extension
# from distutils.core import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import numpy


CURRENT_DIR = os.path.dirname(__file__)
os.chdir(CURRENT_DIR)

def get_version():
    d = {}
    with open("cellcloudx/_version.py") as f:
        exec(f.read(), d)
    return d['__version__'] 

def install_requires(fname="requirements.txt"):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def get_pybind_include():
    try:
        import pybind11
    except ImportError:
        if subprocess.call([sys.executable, '-m', 'pip', 'install', 'pybind11']):
            raise RuntimeError('pybind11 install failed.')
    import pybind11
    return [ pybind11.get_include(), pybind11.get_include(True)]

def find_include_dirs():
    include_dirs = []
    include_dirs += get_pybind_include()
    include_dirs += [numpy.get_include()]
    include_dirs += ['cellcloudx/third_party/eigen3']

    include_dirs += [
        "/usr/local/include/eigen3",
        "/usr/local/homebrew/include/eigen3",
        "/opt/local/var/macports/software/eigen3",
        "/opt/local/include/eigen3",
        "/usr/include/eigen3",
        "/usr/include/local",
        "/usr/include",
    ]
    return include_dirs

__version__ = get_version()
include_dirs=find_include_dirs()

ext_modules=([
    Extension(
        'cellcloudx._ifgt',
        ['cellcloudx/third_party/ifgt_py.cc', 
         'cellcloudx/third_party/ifgt.cc', 
         'cellcloudx/third_party/kcenter_clustering.cc'],

        include_dirs=include_dirs,
        define_macros=[('VERSION_INFO', __version__)],
        language='c++'
    ),
    Extension(
        'cellcloudx._permutohedral_lattice',
        ['cellcloudx/third_party/permutohedral_lattice_py.cc',
         'cellcloudx/third_party/permutohedral.cpp'],
        include_dirs=include_dirs,
        define_macros=[('VERSION_INFO', __version__)],
        language='c++'
    ),
    # Extension(
    #     'cellcloudx.FGT',
    #     ['cellcloudx/third_party/fgt/cluster-openmp.cpp',
    #     'cellcloudx/third_party/fgt/cluster-sequential.cpp',
    #     'cellcloudx/third_party/fgt/cluster.cpp',
    #     'cellcloudx/third_party/fgt/direct.cpp',
    #     'cellcloudx/third_party/fgt/direct_tree.cpp',
    #     'cellcloudx/third_party/fgt/fgt_py.cc',
    #     'cellcloudx/third_party/fgt/ifgt.cpp',
    #     'cellcloudx/third_party/fgt/openmp.cpp',
    #     'cellcloudx/third_party/fgt/transform.cpp',
    #      ],
    #     include_dirs=include_dirs,
    #     define_macros=[('VERSION_INFO', __version__)],
    #     language='c++'
    # ),
])

setup(
    name='cellcloudx',
    version=__version__,
    # packages=['cellcloudx'],
    packages = find_packages(where='.', exclude=(), include=('*',)),
    author='Wei Zhou',
    author_email='welljoea@gmail.com',
    maintainer='Wei Zhou',
    url='',
    description='CellCloudX -a toolkit for analyzing spatial multi-omics data.',
    long_description=open('README.md').read(),
    ext_modules=ext_modules,
    install_requires=install_requires(),
    zip_safe=False,
)