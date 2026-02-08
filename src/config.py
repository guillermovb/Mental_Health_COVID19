import os
from os.path import dirname, join
import sys
import yaml

#_global_configpath = "C:/Users/gvillanueva/Desktop/Mental_Health/config.yaml"
_global_configpath = os.path.abspath(os.path.join(dirname(os.path.realpath(__file__)), os.pardir))
#_global_configpath = join(os.path.join(_global_configpath, os.pardir),"config.yaml")
_global_configpath = join(_global_configpath,"config.yaml")


def get_configpath() -> str:
    return _global_configpath

def get_config() -> dict:
    with open(get_configpath(), 'rt') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


class _Config:
    
    def __init__(self):
        """This method is defined to remind you that this is not a static class"""

    @property
    def path_dataset(self):
        return get_config()['path_dataset']
    
    @property
    def path_codebook(self):
        return get_config()['path_codebook']
    
    @property
    def outcomes(self):
        return get_config()['outcomes']
    
    @property
    def non_relevant_vars(self):
        return get_config()['non_relevant_vars']
    
    @property
    def thr_disc_1(self):
        return get_config()['thr_disc_1']
    
    @property
    def thr_disc_2(self):
        return get_config()['thr_disc_2']
    
    @property
    def additional_categ_var(self):
        return get_config()['additional_categ_var']
    
    @property
    def outer_cv(self):
        return get_config()['outer_cv']
    
    @property
    def inner_cv(self):
        return get_config()['inner_cv']
    
    @property
    def clf_name(self):
        return get_config()['clf_name']

    @property
    def categ_nominal_var(self):
        return get_config()['categ_nominal_var']
    
    @property
    def labels(self):
        return get_config()['labels']
    
    @property
    def n_features(self):
        return get_config()['n_features']
    
    @property
    def algorithm(self):
        return get_config()['algorithm']
    
    @property
    def n_iter(self):
        return get_config()['n_iter']

    @property
    def path_results(self):
        return get_config()['path_results']

CONFIG = _Config()