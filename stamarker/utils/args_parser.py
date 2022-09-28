import argparse
import yaml
import os


def save_yaml(yaml_object, file_path):
    with open(file_path, 'w') as yaml_file:
        yaml.dump(yaml_object, yaml_file, default_flow_style=False)

    print(f'Saving yaml: {file_path}')
    return


def parse_args(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return cfg
