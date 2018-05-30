#!/usr/bin/env python
from setuptools import setup

VERSION = '0.0.1'

long_description = "To be updated"

setup_info = dict(
    # Metadata
    name='mldsa',
    version=VERSION,
    author='John Mucong Ding',
    author_email='mcding@mit.edu',
    maintainer='John Mucong Ding',
    maintainer_email='mcding@mit.edu',
    url='https://github.mit.edu/ALFA-MOOCdb/john_ding_nn',
    download_url='',
    description='A data analytics and visualization toolbox for massive open online courses (MOOCs)',
    long_description=long_description,
    license='MIT',

    # Package info
    zip_safe=True,
    packages=['mldsa'],
    install_requires=[
        'torch==0.4.0',
        'numpy==1.14.2',
        'scipy',
        'pandas==0.22.0',
        'scikit-learn',
        'notebook',
        'jupyter',
        'matplotlib==2.2.2',
        'graphviz',
    ],
    python_requires=">=3.5"
)

setup(**setup_info)