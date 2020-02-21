import os
import io
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = read("README.md")

VERSION = find_version("rkmt", "__init__.py")

requirements = [
    "rknn_toolkit",
]

setup(
    # Metadata
    name="rknn-model-tool",
    version=VERSION,
    author="Yusu Pan",
    author_email="xxdsox@gmail.com",
    url="https://github.com/corenel/rknn-model-tool",
    description="A lightweight library to help with using RKNN models.",
    long_description=readme,
    license="MIT",
    # Package info
    packages=find_packages(exclude=(
        "tests",
        "tests.*",
    )),
    zip_safe=True,
    install_requires=requirements,
)
