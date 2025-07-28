#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : cellcloudx                                   *
* @Author  : Wei Zhou                                     *
* @Date    : 2023/09/27 13:30:28                          *
* @E-mail  : welljoea@gmail.com                           *
* @Version : --                                           *
* You are using the program scripted by Wei Zhou.         *
* Please give me feedback if you find any problems.       *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''

__all__ = [
    'io', 'pl', 'pp', 'rg', 'ag', 'tg', 'nn', 
    'tf', 'ut', 'tl', 'third_party', 'Logger', '__version__'
]

from . import plotting as pl
from . import preprocessing as pp
from . import alignment as ag
from . import transform as tf
from . import utilis as ut
from . import tools as tl
from .io._logger import Logger
from ._version import __version__

def __getattr__(name):
    if name == 'io':
        from . import io
        return io
    # elif name == 'pl':
    #     from . import plotting as pl
    #     return pl
    # elif name == 'pp':
    #     from . import preprocessing as pp
    #     return pp

    # elif name == 'ag':
    #     from . import alignment as ag
    #     return ag
    # elif name == 'tf':
    #     from . import transform as tf
    #     return tf
    # elif name == 'ut':
    #     from . import utilis as ut
    #     return ut
    # elif name == 'tl':
    #     from . import tools as tl
    #     return tl
    elif name == 'tg':
        from . import integration as tg
        return tg
    elif name == 'rg':
        from . import registration as rg
        return rg
    elif name == 'Logger':
        from .io._logger import Logger
        return Logger
    elif name == 'nn':
        from . import nn
        return nn
    elif name == 'third_party':
        from . import third_party
        return third_party
    else:
        raise AttributeError(f"module 'cellcloudx' has no attribute '{name}'")

def __dir__():
    return __all__

# log = Logger('cellcloud3d.log')  #, args.commands

