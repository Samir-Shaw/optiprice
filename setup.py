from setuptools import find_packages, setup

def read_requirements(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        packages = [pkg.strip() for pkg in lines if pkg.strip()]
    return packages

setup(
    name="optiprice",
    version="0.1",
    author="samir",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt")
)