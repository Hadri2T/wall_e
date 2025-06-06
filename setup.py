from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name="Wall-e_package",
      version="0.0.7",
      author="Nathan",
      install_requires=requirements,
      packages=find_packages(),
      include_package_data=True)
