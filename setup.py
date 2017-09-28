#!/usr/bin/env python
import os
from setuptools import setup, find_packages

base = os.path.dirname(os.path.abspath(__file__))

README_PATH = os.path.join(base, "README.rst")

install_requires = [
    "pandas",
    "numpy",
]

tests_require = []

setup(name='anomaly-detection-models',
      version='0.1.1',
      description='',
      long_description=open(README_PATH).read(),
      author='Yun Xu',
      author_email='yunxu1992@gmail.com',
      url='',
      packages=find_packages(),
      install_requires=install_requires,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Operating System :: MacOS',
          'Operating System :: POSIX :: Linux',
          'Topic :: System :: Software Distribution',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
      ],
)
