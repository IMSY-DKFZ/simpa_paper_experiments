import os
import inspect


def get_save_path(experiment_section_name: str, experiment_name: str) -> str:
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
    save_path = os.path.join(base_path, "output_data", experiment_section_name, experiment_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path
