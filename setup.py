from setuptools import setup

setup(
    name='Peony',
    version='0.1dev',
    packages=['peony'],
    license='GPLv3',
    scripts=['bin/peony', 'bin/peony_gee_download', 'bin/peony_success_matrix', 'bin/peony_lcz_inference'],
    long_description=open('README.md').read()
)
