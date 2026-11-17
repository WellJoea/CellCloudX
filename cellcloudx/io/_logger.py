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
import sys

class Logger:
    def __init__(self, filename=None, filemode='w', format='c',
                  level='INFO', pass_pkgs=None,
                  default_pkgs=['pyvista', 'anndata', 'trame-vtk', 'vtk', 
                                'mpl_toolkits', 'plotly', 'harmonypy', 'matplotlib'],
                  **kwargs):
        self.level_dict = {
            'NOTSET': logging.NOTSET,
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }

        if pass_pkgs is not None:
            if isinstance(pass_pkgs, str):
                pass_pkgs = [pass_pkgs]
            default_pkgs += pass_pkgs

        self.Formatter = {
            'c': '[%(asctime)s] [%(levelname)-4s]: %(message)s',
            'p': '[%(levelname)-4s]: %(message)s',
            'n': '%(message)s',
        }

        level = self.level_dict.get(level, level)
        formatter_str = self.Formatter.get(format, format)

        logger_name = __name__ + ".custom_logger"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        
        self.logger.propagate = False
        
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        formatter = logging.Formatter(formatter_str, datefmt='%Y-%m-%d %H:%M:%S')
        
        if filename:
            file_handler = logging.FileHandler(filename, mode=filemode)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(level)
            self.logger.addHandler(console_handler)
        
        self.silence_loggers(default_pkgs)

    def silence_loggers(self, silence_pkgs=['pyvista', 'anndata']):
        for pkg in silence_pkgs:
            try:
                pkg_logger = logging.getLogger(pkg)
                pkg_logger.setLevel(logging.CRITICAL + 1000)
                pkg_logger.propagate = False
            except Exception:
                pass


_logger = Logger()
logger  = _logger.logger

def Setting(*args, **kwargs):
    default_args = {
        'filename': None,
        'filemode': 'w',
        'format': 'c',
        'level': 'INFO',
        'rlevel': 'DEBUG',
    }
    default_args.update(kwargs)
    
    global _logger, logger
    _logger = Logger(**default_args)
    logger  = _logger.logger

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