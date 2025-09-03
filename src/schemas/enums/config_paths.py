from enum import Enum
import os


def find_project_root(start_path):
    """
    Finds the project root directory by searching for a specific marker file.
    """
    current_path = os.path.abspath(start_path)
    while True:
        if os.path.exists(os.path.join(current_path, 'configs')):
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            # Reached the root of the file system
            raise FileNotFoundError(
                "Could not find project root containing 'configs' directory.")
        current_path = parent_path


CONFIG_DIR = "configs"


class ConfigName(str, Enum):
    DEFAULT = "default.yaml"
    EXPERIMENT = "experiment.yaml"
    DEBUG = "debug.yaml"

    def path(self) -> str:
        """Return full path to config file"""
        try:
            project_root = find_project_root(__file__)
            complete_config_path = os.path.join(
                project_root, CONFIG_DIR, self.value)
            return complete_config_path
        except FileNotFoundError as e:
            raise RuntimeError(
                "Project root could not be determined. Ensure you are running the script within the project directory."
            ) from e
