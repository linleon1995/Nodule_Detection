import argparse
import yaml


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))


class DictAsMember(dict):
    def __setattr__(self, __name, __value) -> None:
        self.__setitem__(__name, __value)

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


def load_config(config_reference=None, dict_as_member=False):
    if isinstance(config_reference, str):
        parser = argparse.ArgumentParser(description='DL')
        if config_reference is not None:
            parser.add_argument('--config', type=str, help='Path to the YAML config file', default=config_reference)
        else:
            parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
        args = parser.parse_args()
        config = _load_config_yaml(args.config)
    elif isinstance(config_reference, dict):
        config = config_reference

    if dict_as_member:
        config = DictAsMember(config)
    return config