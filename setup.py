from setuptools import setup, find_packages

setup(
    name='deform_rl',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
    ]
)
