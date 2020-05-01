# Generic utility functions
import os
import re
import fnmatch


def get_splunkhome_path():
    return os.path.normpath(os.environ['SPLUNK_HOME'])

def get_etc_path():
    return os.environ.get('SPLUNK_ETC', os.path.join(get_splunkhome_path(), 'etc'))

def make_splunkhome_path(p):
    return os.path.join(get_splunkhome_path(), *p)

def get_apps_path():
    return os.path.normpath(os.path.join(get_etc_path(), 'apps'))

def get_mltk_bin_path():
    return os.path.normpath(os.path.join(get_etc_path(), 'apps', 'Splunk_ML_Toolkit', 'bin'))

def get_spool_path(filename):
    return os.path.normpath(os.path.join(get_splunkhome_path(), 'var', 'spool', 'splunk', filename))
