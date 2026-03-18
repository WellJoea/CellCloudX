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
import sys
import tempfile
import shutil
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
import distutils.sysconfig

# 使用绝对路径避免chdir
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_version():
    version_file = os.path.join(CURRENT_DIR, "cellcloudx", "_version.py")
    with open(version_file) as f:
        # 使用更安全的方式获取版本
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

def install_requires(fname="requirements.txt"):
    req_file = os.path.join(CURRENT_DIR, fname)
    if not os.path.exists(req_file):
        return []
    
    with open(req_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def get_pybind_include():
    try:
        import pybind11
    except ImportError:
        # 使用当前Python解释器安装
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pybind11'])
        import pybind11
    return [pybind11.get_include(), pybind11.get_include(True)]

def find_include_dirs():
    include_dirs = []
    
    # 添加pybind11路径
    include_dirs.extend(get_pybind_include())
    
    # 添加numpy路径（延迟导入）
    try:
        import numpy
        include_dirs.append(numpy.get_include())
    except ImportError:
        pass
    
    # 添加本地第三方库路径
    eigen_local = os.path.join(CURRENT_DIR, "cellcloudx", "third_party", "eigen3.4")
    if os.path.exists(eigen_local):
        include_dirs.append(eigen_local)
    
    # 添加系统级Eigen路径
    eigen_paths = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/opt/local/include/eigen3",
        "/usr/local/homebrew/include/eigen3",
        "/opt/homebrew/include/eigen3",
        "/opt/local/var/macports/software/eigen3",
    ]
    
    # 添加环境变量指定的路径
    if 'EIGEN3_INCLUDE_DIR' in os.environ:
        include_dirs.append(os.environ['EIGEN3_INCLUDE_DIR'])
    
    # 添加其他环境变量路径
    for env_var in ['C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH']:
        if env_var in os.environ:
            include_dirs.extend(os.environ[env_var].split(':'))
    
    # 添加有效的系统路径
    include_dirs.extend(path for path in eigen_paths if os.path.exists(path))
    
    # 去重
    return list(set(include_dirs))

def _check_for_openmp():
    """更高效的OpenMP检测"""
    test_code = """
#include <omp.h>
int main() {
    #pragma omp parallel
    omp_get_thread_num();
    return 0;
}
"""
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(test_code)
            f.flush()
            
            # 尝试编译测试代码
            result = subprocess.run(
                [os.environ.get('CC', 'cc'), '-fopenmp', '-o', '/dev/null', f.name],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
            return result.returncode == 0
    finally:
        if os.path.exists(f.name):
            os.unlink(f.name)
    return False

def get_extensions():
    """按需构建扩展模块"""
    # 只有在需要编译时才收集头文件路径
    include_dirs = find_include_dirs()
    has_openmp = _check_for_openmp()
    
    common_args = {
        'include_dirs': include_dirs,
        'language': 'c++',
        'extra_compile_args': ['-O3', '-march=native', '-mtune=native', '-ffast-math'],
    }
    
    if has_openmp:
        common_args['extra_link_args'] = ['-lgomp']
        common_args['extra_compile_args'].append('-fopenmp')
    
    extensions = [
        Extension(
            'cellcloudx.third_party._ifgt',
            sources=[
                os.path.join('cellcloudx', 'third_party', 'ifgt', 'ifgt_py.cc'),
                os.path.join('cellcloudx', 'third_party', 'ifgt', 'ifgt.cc'),
                os.path.join('cellcloudx', 'third_party', 'ifgt', 'kcenter_clustering.cc')
            ],
            **common_args
        ),
        Extension(
            'cellcloudx.third_party._permutohedral_lattice',
            sources=[
                os.path.join('cellcloudx', 'third_party', 'others', 'permutohedral_lattice_py.cc'),
                os.path.join('cellcloudx', 'third_party', 'others', 'permutohedral.cpp')
            ],
            **common_args
        ),
    ]
    
    return extensions

class OptimizedBuild(build_ext):
    """优化的构建扩展"""
    def build_extensions(self):
        # 设置C++标准
        cpp_std = self.detect_cpp_std()
        
        for ext in self.extensions:
            # 添加平台特定优化
            if self.compiler.compiler_type == 'unix':
                ext.extra_compile_args.extend([
                    cpp_std,
                    '-fvisibility=hidden',
                    '-funroll-loops',
                    '-flto'  # 链接时优化
                ])
                
                if sys.platform == 'darwin':
                    ext.extra_compile_args.extend([
                        '-stdlib=libc++',
                        '-mmacosx-version-min=10.15'
                    ])
            
            # 添加Windows特定选项
            elif self.compiler.compiler_type == 'msvc':
                ext.extra_compile_args.extend([
                    '/O2',   # 最大优化
                    '/GL',    # 全程序优化
                    '/arch:AVX2' if sys.version_info >= (3, 9) else '/arch:AVX'
                ])
        
        super().build_extensions()
    
    def detect_cpp_std(self):
        """自动检测支持的最高C++标准"""
        test_stds = [
            ('-std=c++20', 'c++20'),
            ('-std=c++17', 'c++17'),
            ('-std=c++14', 'c++14'),
            ('-std=c++11', 'c++11')
        ]
        
        test_code = "int main() { return 0; }"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(test_code)
            f.flush()
            
            for flag, std_name in test_stds:
                try:
                    subprocess.check_call(
                        [self.compiler.compiler[0], flag, '-c', f.name, '-o', os.devnull],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print(f"Using C++ standard: {std_name}")
                    return flag
                except subprocess.CalledProcessError:
                    continue
        
        # 默认使用C++11
        print("Warning: Using default C++11 standard")
        return '-std=c++11'

# 获取版本
__version__ = get_version()

# 读取README内容
readme_path = os.path.join(CURRENT_DIR, 'README.md')
long_description = ''
if os.path.exists(readme_path):
    with open(readme_path, encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='cellcloudx',
    version=__version__,
    packages=find_packages(where='.', exclude=('tests', 'docs', 'examples')),
    author='Wei Zhou',
    author_email='welljoea@gmail.com',
    maintainer='Wei Zhou',
    url='https://github.com/WellJoea/CellCloudX.git',
    description='CellCloudX - Advanced toolkit for spatial multi-omics data analysis',
    long_description=long_description,
    # long_description_content_type='text/markdown',
    zip_safe=False,
    install_requires=install_requires(),
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    ext_modules=get_extensions(),
    cmdclass={'build_ext': OptimizedBuild},
    package_data={
        'cellcloudx': [
            'third_party/**/*.h',
            'third_party/**/*.hpp'
        ]
    },
    include_package_data=True,
    setup_requires=['numpy', 'pybind11'] if get_extensions() else [],
)