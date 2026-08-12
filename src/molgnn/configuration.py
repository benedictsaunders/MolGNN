import json
from typing import Any
from pathlib import Path

class Configuration:
    """
    Global configuration state, allowing access to settings at the class level across the project.
    """

    _filepath = "configuration.json"
    _config = dict()
    _loaded = False

    @classmethod
    def set_filepath(cls, filepath: str | Path = "configuration.json") -> None:
        """
        Update the filepath of the config file for saving/loading

        Args:
            filepath (str or Path): Come on...
        """
        cls._filepath = str(filepath)
        cls.load()

    @classmethod
    def load(cls) -> None:
        """
        Read the config file into memory
        """
        print(Path(cls._filepath))
        if not Path(cls._filepath).exists():
            cls._config = dict()
            cls._loaded = True
            return None

        try:
            with open(cls._filepath, "r") as f:
                cls._config = json.load(f)
            cls._loaded = True
        except json.JSONDecodeError as v_e:
            raise ValueError(f"Error in reading {cls._filepath}: {v_e}")
        except IOError as i_e:
            raise IOError(f"Failed to read {cls._filepath}: {i_e}")

    @classmethod
    def save(cls) -> None:
        """
        Write in-memory configuration object to disk.
        """
        try:
            with open(cls._filepath, "w") as f:
                json.dump(cls._config, f, indent=4)
        except IOError as i_e:
            raise IOError(f"Failed to write {cls._filepath}: {i_e}")

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Getter function, loads automatically on firt use.

        Args:
            key (str): Configuration key

        Returns:
            Any: configuration setting at `key`
        """
        if not cls._loaded:
            cls.load()
        return cls._config.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any, auto_save: bool = False) -> None:
        """
        Sets a value in memory, optionally writing to disk.

        Args:
            key (str): Configuration key
            value (Any): value to be assigned to `key` setting
        """
        if not cls._loaded:
            cls.load()
        cls._config[key] = value
        if auto_save:
            cls.save()
