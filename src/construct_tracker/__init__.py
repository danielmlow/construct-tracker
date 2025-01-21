"""Init file"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("construct-tracker")
except PackageNotFoundError:
    __version__ = "unknown"

# from construct_tracker.load_datasets import load_data

# __all__ = ["load_data"]
