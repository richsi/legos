import yaml

def load_config(config_file, config_name):
  with open(config_file, "r") as f:
    all_configs = yaml.safe_load(f)
    assert(all_configs.get(config_name, {}) != {})
  return all_configs.get(config_name, {})