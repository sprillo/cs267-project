from setuptools import setup, find_packages


# Import version
__builtins__.__CS267_PROJECT_SETUP__ = True
from src import __version__ as version


setup(
    name='cs267_project',
    version=version,
    packages=find_packages(),
    install_requires=[
        'ete3',
        'flake8',
        'networkx',
        'numpy',
        'pandas',
        'parameterized',
        'pytest',
        'threadpoolctl',
        'torch',
        'tqdm',
    ],
)
