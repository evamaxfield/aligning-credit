"""Top-level package for aligning-credit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aligning-credit")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown"
__email__ = "evamaxfieldbrown@gmail.com"
