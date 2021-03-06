import yaml
from typing import Dict

required_keys = [
    'app_name',
    'image_dir',
    'job_dir',
    'cam_num',
    'always_on_top',
    'update_period_ms',
    'present_timer_sec',
    'presence_image_dir',
    'presence_job_dir',
]


def load_yaml(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        yml = yaml.safe_load(f)

        for key in required_keys:
            if key not in yml.keys():
                raise ValueError(f'missing config key: {key}')

        return yml
