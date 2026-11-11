# This file is vendored from ivadomed 2.9.10 (https://github.com/ivadomed/ivadomed),
# originally `ivadomed/config_manager.py`, licensed under the MIT License.
# See the LICENSE.md file in this directory for the full license text.
#
# Modifications for SCT:
#   - Fixed flake8 warnings, without changing behavior.
#   - Imports point to `spinalcordtoolbox.ivadomed` instead of `ivadomed`.

import json
import collections.abc
from typing import Dict, List, Any, Union

from loguru import logger
from pathlib import Path
from spinalcordtoolbox.ivadomed import utils as imed_utils
from spinalcordtoolbox.ivadomed.keywords import ConfigKW, LoaderParamsKW, SplitDatasetKW, DataTestingKW
import copy


def update(source_dict: dict, destination_dict: dict) -> dict:
    """Update dictionary and nested dictionaries.

    Args:
        source_dict (dict): Source dictionary that is updated by destination dictionary.
        destination_dict (dict): Destination dictionary.

    Returns:
        dict: updated dictionary
    """
    for key, value in destination_dict.items():
        if isinstance(value, collections.abc.Mapping):
            source_dict[key] = update(source_dict.get(key, {}), value)
        else:
            # If source dictionary has keys that the destination dict doesn't have, keep these keys
            if key in source_dict and isinstance(source_dict[key], collections.abc.Mapping) and not isinstance(value,
                                                                                                               collections.abc.Mapping):
                pass
            else:
                source_dict[key] = value
    return source_dict


def deep_dict_compare(source_dict: dict, destination_dict: dict, keyname: str = None):
    """Compare and display differences between dictionaries (and nested dictionaries).

    Args:
        source_dict (dict): Source dictionary.
        destination_dict (dict): Destination dictionary.
        keyname (str): Key name to indicate the path to nested parameter.

    """
    for key in destination_dict:
        if key not in source_dict:
            key_str = key if keyname is None else keyname + key
            logger.info(f'    {key_str}: {destination_dict[key]}')

        else:
            if isinstance(destination_dict[key], collections.abc.Mapping):
                if isinstance(source_dict[key], collections.abc.Mapping):
                    deep_dict_compare(source_dict[key], destination_dict[key], key + ": ")
                # In case a new dictionary appears in updated file
                else:
                    deep_dict_compare(source_dict, destination_dict[key], key + ": ")


def load_json(config_path: str) -> dict:
    """Load json file content

    Args:
        config_path (str): Path to json file.

    Returns:
        dict: config dictionary.

    """
    with open(config_path, "r") as fhandle:
        default_config = json.load(fhandle)
    return default_config


# To ensure retro-compatibility for parameter changes in configuration file
KEY_CHANGE_DICT = {'UNet3D': ConfigKW.MODIFIED_3D_UNET, 'bids_path': LoaderParamsKW.PATH_DATA,
                   'log_directory': ConfigKW.PATH_OUTPUT}
KEY_SPLIT_DATASET_CHANGE_LST = ['method', 'center_test']


class ConfigurationManager(object):
    """Configuration file manager.

    Args:
        path_context (str): Path to configuration file.
    Attributes:
        path_context (str): Path to configuration file.
        config_default (dict): Default configuration file from ``ivadomed`` package.
        context_original (dict): Provided configuration file.
        config_updated (dict): Updated configuration file.
    """

    def __init__(self, path_context: str):
        """
        Initialize the ConfigurationManager by validating the given path and loading the file.
        Also load the default configuration file.

        Args:
            path_context (str): Path to configuration file.
        """
        self.path_context: str = path_context
        self.key_change_dict: Dict[str, str] = KEY_CHANGE_DICT
        self.key_split_dataset_change_lst: List[str] = KEY_SPLIT_DATASET_CHANGE_LST
        self._validate_path()
        default_config_path: str = str(Path(imed_utils.__ivadomed_dir__, "ivadomed", "config", "config_default.json"))
        self.config_default: dict = load_json(default_config_path)
        self.context_original: dict = load_json(path_context)
        self.config_updated: dict = {}

    @property
    def config_updated(self) -> dict:
        """
        This function simply returns the attribute `_config_updated`.

        Returns:
            dict: `_config_updated` attribute
        """
        return self._config_updated

    @config_updated.setter
    def config_updated(self, config_updated: dict):
        """
        If config_updated is empty we copy the loaded configuration into it and apply some changes (changing keys name,
        changing values, deleting key-value pair) to ensure retro-compatibility.
        Sets the new config_updated to the attribute `_config_updated`.

        Args:
            config_updated (dict): The new configuration to set.
        """
        if config_updated == {}:
            context: dict = copy.deepcopy(self.context_original)
            self.change_keys(context, list(context.keys()))
            config_updated: dict = update(self.config_default, context)
            self.change_keys_values(config_updated[ConfigKW.SPLIT_DATASET],
                                    config_updated[ConfigKW.SPLIT_DATASET].keys())

        self._config_updated: dict = config_updated
        if config_updated['debugging']:
            self._display_differing_keys()

    def get_config(self) -> dict:
        """Get updated configuration file with all parameters from the default config file.

        Returns:
            dict: Updated configuration dict.
        """
        return self.config_updated

    def change_keys(self, context: Union[dict, collections.abc.Mapping], keys: List[str]):
        """
        This function changes the name of the keys of the context dictionary, that are also in the `key_change_dict`
        attribute, to the values that are associated with them in the `key_change_dict` attribute.

        Args:
            context (Union[dict, collections.abc.Mapping]): The dictionary to change.
            keys (List[str]): The keys in context to consider.
        """
        for key_to_change in keys:
            # Verify if key is still in the dict
            if key_to_change in context:
                # If the key_to_change is "NumpyToTensor", remove it from the context.
                if key_to_change == "NumpyToTensor":
                    del context[key_to_change]
                    continue
                value_to_change: Any = context[key_to_change]
                # Verify if value is a dictionary
                if isinstance(value_to_change, collections.abc.Mapping):
                    self.change_keys(value_to_change, list(value_to_change.keys()))
                else:
                    # Change keys from the key_change_dict
                    for key in self.key_change_dict:
                        if key in context:
                            context[self.key_change_dict[key]] = context[key]
                            del context[key]

    def change_keys_values(self, config_updated: dict, keys: List[str]):
        """
        This function sets DATA_TESTING->DATA_TYPE to "institution_id" if method value is per_center,
        DATA_TESTING->DATA_VALUE to the value of center_test.
        It is basically verifying some conditions and set values to the `config_updated`.

        Args:
            config_updated (dict): Configuration dictionary to update.
            keys (List[str]): The keys to consider.
        """
        for key_to_change in self.key_split_dataset_change_lst:
            if key_to_change in keys:
                value: Any = config_updated[key_to_change]
                # If the method is per_center, the data_testing->data_type value becomes "institution_id".
                if key_to_change == 'method' and value == "per_center":
                    config_updated[SplitDatasetKW.DATA_TESTING][DataTestingKW.DATA_TYPE] = "institution_id"
                # If [the key is center_test], [data_testing->data_type == "institution_id"] and [the value is not None]
                # data_testing->data_type value becomes value of config_updated
                if key_to_change == 'center_test' and \
                        config_updated[SplitDatasetKW.DATA_TESTING][DataTestingKW.DATA_TYPE] == "institution_id" and \
                        value is not None:
                    config_updated[SplitDatasetKW.DATA_TESTING][DataTestingKW.DATA_VALUE] = value
                # Remove the value of the current key
                del config_updated[key_to_change]

    def _display_differing_keys(self):
        """Display differences between dictionaries.
        """
        logger.info('Adding the following keys to the configuration file')
        deep_dict_compare(self.context_original, self.config_updated)
        logger.info('\n')

    def _validate_path(self):
        """Ensure validity of configuration file path.
        """
        if not Path(self.path_context).exists():
            raise ValueError(
                f"\nERROR: The provided configuration file path (.json) does not exist: "
                f"{Path(self.path_context).absolute()}\n")
        elif Path(self.path_context).is_dir():
            raise IsADirectoryError(f"ERROR: The provided configuration file path (.json) is a directory not a file: "
                                    f"{Path(self.path_context).absolute()}\n")
        elif not Path(self.path_context).is_file():
            raise FileNotFoundError(f"ERROR: The provided configuration file path (.json) is not found: "
                                    f"{Path(self.path_context).absolute()}\n")
        elif self.path_context.endswith('.yaml') or self.path_context.endswith('.yml'):
            raise ValueError(
                f"\nERROR: The provided configuration file path (.json) is a yaml file not a json file, "
                f"yaml files are not yet supported: {Path(self.path_context).absolute()}\n")
        elif not self.path_context.endswith('.json'):
            raise ValueError(
                f"\nERROR: The provided configuration file path (.json) is not a json file: "
                f"{Path(self.path_context).absolute()}\n")
