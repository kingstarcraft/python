from setuptools import setup, find_packages

setup(
    name='zero',
    version='1.6.2',
    description='util of python',
    author='xi long',
    author_email='kingstarcraft@foxmail.com',
    install_requires=['torch', 'sklearn'],
    packages=find_packages()
)
