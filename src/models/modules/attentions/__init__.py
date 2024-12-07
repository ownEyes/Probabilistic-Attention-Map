import importlib

import torch.nn as nn

# from .pdfam import *
# from .lap_experiments import *


def find_module_using_name(module_name):

    if module_name == "none":
        return None

    module_base_path = "models.modules.attentions"
    module_filename = module_name + "_module"
    if 'pdfam' in module_name:
        module_filename = module_base_path+'.pdfam.' + module_filename
    elif 'lap' in module_name:
        module_filename = module_base_path+'.lap_experiments.' + module_filename
    else:
        module_filename = module_base_path+'.' + module_filename
    modellib = importlib.import_module(
        module_filename)

    module = None
    target_model_name = module_name + '_module'

    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, nn.Module):
            module = cls

    if module is None:
        print("In %s.py, there should be a subclass of nn.Module with class name that matches %s in lowercase." % (
            module_filename, target_model_name))
        exit(0)

    return module


def get_attention_module(attention_type="none"):

    return find_module_using_name(attention_type.lower())
