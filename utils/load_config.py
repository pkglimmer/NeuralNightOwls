import yaml

class Config:
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                # Convert nested dictionaries to Config objects as well
                self.__dict__[key] = Config(**value)
            else:
                self.__dict__[key] = value
    def get_dict(self):
        return self.__dict__

class Config_dict:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
    def get_dict(self):
        return self.__dict__

def load_config(config_file):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)

def load_config_as_dict(config_file):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config_dict(**config_dict)

# Usage
# config = load_config('utils/config_dagmm.yaml')
# print(config.dirs.data_dir)  # Using attribute access
