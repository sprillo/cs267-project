from setuptools import setup, find_packages


# Import version
__builtins__.__CHERRYML_SETUP__ = True
from cherryml import __version__ as version


setup(
    name='cs267_project',
    version=version,
    packages=find_packages(),
    install_requires=[
        'biotite',
        'ete3',
        'flake8',
        'json',
        'matplotlib',
        'networkx',
        'numpy',
        'pandas',
        'parameterized',
        'pytest',
        'scipy',
        'seaborn',
        'threadpoolctl',
        'torch',
        'tqdm',
        'wget',
    ],
)
