import importlib
import torch.nn as nn


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

    
def get_model(model_config):
    def _model_class(class_name):
        modules = ['model']
        for module in modules:
            m = importlib.import_module(module)
            clazz = getattr(m, class_name, None)
            if clazz is not None:
                return clazz

    model_class = _model_class(model_config['name'])
    return model_class(**model_config)


def get_activation(name, *args, **kwargs):
    if name == 'relu':
        return nn.ReLU(inplace=True, *args, **kwargs)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'softmax':
        return nn.Softmax(*args, **kwargs)