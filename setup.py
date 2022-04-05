from setuptools import setup

setup(
    name='Peony',
    version='0.1dev',
    packages=['peony'],
    license='GPLv3',
    scripts=['bin/peony'],
    long_description=open('README.md').read()
)
