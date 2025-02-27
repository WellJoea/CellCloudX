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
from setuptools import find_packages, setup, Extension
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
    return [pybind11.get_include(), pybind11.get_include(True)]

def find_include_dirs():
    include_dirs = []
    include_dirs += get_pybind_include()
    include_dirs += [numpy.get_include()]
    eigen3_env = os.getenv('EIGEN3_INCLUDE_DIR')
    if eigen3_env:
        include_dirs.append(eigen3_env)
    include_dirs += ['cellcloudx/third_party/eigen3.4']

    include_dirs += [
        # "/usr/include/x86_64-linux-gnu",  
        # "/usr/include/x86_64-linux-gnu/bits",
        # "/usr/include",
        # "/usr/include/local",
        # "/usr/local/include",
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/opt/local/include/eigen3",
        "/usr/local/homebrew/include/eigen3",
        "/opt/homebrew/include/eigen3",
        "/opt/local/var/macports/software/eigen3", 
    ]
    include_dirs += os.getenv('C_INCLUDE_PATH', '').split(':')
    include_dirs += os.getenv('CPLUS_INCLUDE_PATH', '').split(':')
    
    include_dirs = [p for p in include_dirs if p]
    print("Final include paths:\n", include_dirs)
    return include_dirs

def _check_for_openmp():
    """Check OpenMP."""
    import distutils.sysconfig
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix='cellcloudx')
    compiler = os.environ.get('CC', distutils.sysconfig.get_config_var('CC'))
    if compiler is None:
        return False
    compiler = compiler.split()[0]

    # Attempt to compile a test script.
    # See http://openmp.org/wp/openmp-compilers/
    tmpfile = os.path.join(tmpdir, 'check_openmp.c')
    with open(tmpfile, 'w') as f:
        f.write('''
#include <omp.h>
#include <stdio.h>
int main() {
    #pragma omp parallel
    printf("Hello from thread %d", omp_get_thread_num());
}
''')

    try:
        with open(os.devnull, 'w') as fnull:
            exit_code = subprocess.call(
                [compiler, '-fopenmp', '-o%s' % os.path.join(tmpdir, 'check_openmp'), tmpfile],
                stdout=fnull, stderr=fnull
            )
    except OSError:
        print ('Suggestion: You should build using OpenMP...\n')
        exit_code = 1
    finally:
        shutil.rmtree(tmpdir)

    if exit_code == 0:
        print ('Continuing your build using OpenMP...\n')
        return True
    return False


__version__ = get_version()
include_dirs=find_include_dirs()

ext_modules = ([
    Extension(
        'cellcloudx.third_party._ifgt',
        [
            'cellcloudx/third_party/ifgt/ifgt_py.cc',
            'cellcloudx/third_party/ifgt/ifgt.cc',
            'cellcloudx/third_party/ifgt/kcenter_clustering.cc'
        ],
        include_dirs=include_dirs,
        extra_link_args=['-lgomp'] if _check_for_openmp() else [],
        define_macros=[('VERSION_INFO', __version__)],
        language='c++'
    ),
    Extension(
        'cellcloudx.third_party._permutohedral_lattice',
        [
            'cellcloudx/third_party/others/permutohedral_lattice_py.cc',
            'cellcloudx/third_party/others/permutohedral.cpp'
        ],
        include_dirs=include_dirs,
        extra_link_args=['-lgomp'] if _check_for_openmp() else [],
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


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!\n'
                           'conda install -c conda-forge gcc_linux-64=14.2 gxx_linux-64=14.2  gfortran_linux-64 libgcc-ng\n'
                           'conda list |egrep "gcc|gxx"\n'
                           )
class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': ['-mtune=native'], #['-march=native', '-mtune=native']
    }
    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if _check_for_openmp():
                opts.append('-fopenmp')
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='cellcloudx',
    version=__version__,
    # packages=['cellcloudx'],
    packages=find_packages(where='.', exclude=(), include=('*',)),
    author='Wei Zhou',
    author_email='welljoea@gmail.com',
    maintainer='Wei Zhou',
    url='',
    description='CellCloudX - a toolkit for analyzing spatial multi-omics data.',
    long_description=open('README.md').read(),
    ext_modules=ext_modules,
    install_requires=install_requires(),
    zip_safe=False,
    cmdclass={'build_ext': BuildExt},
)

#conda install -c conda-forge gcc_linux-64=14.2   gxx_linux-64=14.2  gfortran_linux-64 libgcc-ng
#conda list |grep gcc

# 注意：
# 如果编译过程中仍然出现 "Eigen/Core: 没有那个文件或目录" 的错误，
# 请确保已安装 Eigen 库（例如，在 Ubuntu 上执行：
#     sudo apt-get install libeigen3-dev
# ）或手动设置环境变量 EIGEN3_INCLUDE_DIR 指定 Eigen 的安装路径，例如：
#     export EIGEN3_INCLUDE_DIR=/path/to/eigen3
