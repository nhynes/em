"""A setuptools based setup module."""

from setuptools import setup, find_packages
import os
from os import path

base_dir = path.abspath(path.dirname(__file__))

with open(os.path.join(base_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='em',

    version='1.0.0',

    description='An tool for managing deep learning experiments.',
    long_description=long_description,

    url='https://github.com/nhynes/em',

    author='Nick Hynes',
    author_email='nhynes@nhynes.com',

    # Choose your license
    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='deep learning pytorch',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=[
        'pygit2>=0.26.0',
        'python-daemon',
    ],

    entry_points={
        'console_scripts': [
            'em=em.__main__:main',
        ],
    },
)
