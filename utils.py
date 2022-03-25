import yaml

__all__ = ['load_yaml_file']


# helper functions and classes
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_yaml_file(file_name):
    with open(file_name) as file:
        doc_dict = yaml.full_load(file)
    return DotDict(doc_dict)
