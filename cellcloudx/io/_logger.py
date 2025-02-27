#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
***********************************************************
* Author  : Zhou Wei                                      *
* Date    : 2020/09/09 10:47:17                           *
* E-mail  : welljoea@gmail.com                            *
* Version : --                                            *
* You are using the program scripted by Zhou Wei.         *
* If you find some bugs, please                           *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''

import logging
class Logger:
    def __init__(self, filename=None, filemode='w', format='c',
                  level = 'INFO', pass_pkgs=None,
                  defualt_pkgs = ['pyvista', 'anndata', 'trame-vtk', 'vtk', 
                                  'mpl_toolkits','plotly' 'harmonypy', 'matplotlib'],
                  **kargs):
        self.level_dict = {
            'NOTSET'  : logging.NOTSET,
            'DEBUG'   : logging.DEBUG,
            'INFO'    : logging.INFO,
            'WARNING' : logging.WARNING,
            'ERROR'   : logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }

        if not pass_pkgs is None:
            if type(pass_pkgs) is str:
                pass_pkgs = [pass_pkgs]
            defualt_pkgs += pass_pkgs

        self.Formatter ={ 
            'c' : '[%(asctime)s] [%(levelname)-4s]: %(message)s',
            'p' : '[%(levelname)-4s]: %(message)s',
            'n' : '%(message)s' ,
        }

        level = self.level_dict.get(level, level)
        former = self.Formatter.get(format, format)

        logger = logging.getLogger(__name__)
        logging.basicConfig(
            level    = level,
            format   = former,
            datefmt  = '%Y-%m-%d %H:%M:%S',
            filename = filename,
            filemode = filemode
        )
        # ch = logging.StreamHandler() # duplicates
        logger.setLevel(level)
        self.pass_logger(defualt_pkgs)
        self.logger = logger

    def pass_logger(self, pass_pkgs=['pyvista', 'anndata']):
        for pkg in pass_pkgs:
            pp = logging.getLogger(pkg)
            pp.setLevel(logging.CRITICAL + 1000)
            pp.propagate = False

logger = Logger().logger
def Setting(*arg, **kargs): # TODO
    defualt_args = {
        'filename': None,
        'filemode': 'w',
        'format'  : 'c',
        'level'   : 'INFO',
        'rlevel'  : 'DEBUG',
    }
    defualt_args.update(kargs)

    global logger
    logger = Logger(defualt_args).logger

'''
import logging
class DispatchingFormatter:
    def __init__(self, formatters, default_formatter):
        self._formatters = formatters
        self._default_formatter = default_formatter

    def format(self, record):
        formatter = self._formatters.get(record.name, self._default_formatter)
        return formatter.format(record)

class Logger:
    level_dict = {
        'NOTSET'  : logging.NOTSET,
        'DEBUG'   : logging.DEBUG,
        'INFO'    : logging.INFO,
        'WARNING' : logging.WARNING,
        'ERROR'   : logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    level_int ={
        0: logging.NOTSET,
        1: logging.DEBUG,
        2: logging.INFO,
        3: logging.WARNING,
        4: logging.ERROR,
        5: logging.CRITICAL,
    }
    ChangeFrom = DispatchingFormatter(
            { 'c' : logging.Formatter( '[%(asctime)s] [%(levelname)-4s]: %(message)s', '%Y-%m-%d %H:%M:%S'),
              'p' : logging.Formatter( '[%(levelname)-4s]: %(message)s'),
              'n' : logging.Formatter( '%(message)s' ),
            }, 
            logging.Formatter('%(message)s')
     )

    def __init__(self, outpath=None, filemode='w',  level = 'INFO', flevel = None):
        if flevel is None:
            flevel = level
        level = Logger.level_dict[level] if isinstance(level, str) else Logger.level_int[level]
        logging.basicConfig(
            level    = level,
            format   = '[%(asctime)s] [%(levelname)-4s]: %(message)s',
            datefmt  = '%Y-%m-%d %H:%M:%S',
            filename = None,
        )

        if outpath is not None:
            level = Logger.level_dict[level] if isinstance(level, str) else Logger.level_int[level]
            File = logging.FileHandler(outpath,  mode= filemode)
            File.setLevel(level)
            File.setFormatter(Logger.ChangeFrom)
            logging.getLogger().addHandler(File)

        self.R = logging
        self.C = logging.getLogger('c')
        self.P = logging.getLogger('p')
        self.N = logging.getLogger('n')
        self.CI = logging.getLogger('c').info
        self.NI = logging.getLogger('n').info
        self.CW = logging.getLogger('c').warning
        self.NW = logging.getLogger('n').warning
'''