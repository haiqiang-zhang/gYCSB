import yaml
from typing import Dict, Any
import os

SYSTEM_SETTING_PATH = "gycsb_setting.yaml"

def __load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        yaml_content = f.read()
        yaml_content = yaml_content.replace('\t', ' ')
        return yaml.safe_load(yaml_content)

def __merge_configs(global_config: Dict[str, Any], workload_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge global and workload configurations, with workload taking precedence."""
    merged = global_config.copy()
    merged.update(workload_config)
    return merged


def get_available_workloads():
    config_path = os.path.join(os.path.dirname(__file__), "..", "workload_config.yaml")
    config = __load_config(config_path)
    return list(config['workloads'].keys())

def get_workload_config(workload_name: str=None):
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "..", "workload_config.yaml")
    config = __load_config(config_path)
    
    if workload_name is None:
        return config['global']
    
    if workload_name not in config['workloads']:
        raise ValueError(f"Workload {workload_name} not found in config")
    
    # Merge configurations
    final_config = __merge_configs(config['global'], config['workloads'][workload_name])
    
    # Display final configuration
    print(f"Selected Configuration: {workload_name}")
    print("-" * 50)
    print(f"Workload: {config['workloads'][workload_name]['name']}")
    print("\nFinal Settings:")
    for key, value in final_config.items():
        print(f"{key}: {value}")
    print("-" * 50)
    
    return final_config


def get_binding_config(binding_name: str):
    config_path = os.path.join(os.path.dirname(__file__), "..", "binding_config.yaml")
    config = __load_config(config_path)
    return config[binding_name]



def load_system_setting():
    path = os.path.join(os.path.dirname(__file__), "..", SYSTEM_SETTING_PATH)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f.read())

def save_system_setting(config):
    # Load existing config
    existing_config = load_system_setting()
    # Update only the provided fields
    existing_config.update(config)
    # Save the merged config
    path = os.path.join(os.path.dirname(__file__), SYSTEM_SETTING_PATH)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(existing_config, f, allow_unicode=True, default_flow_style=False)
