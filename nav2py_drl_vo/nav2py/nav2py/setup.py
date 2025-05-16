#!/usr/bin/env python

from setuptools import setup, find_packages

PROJECT = 'nav2py'

setup(name=PROJECT,
      version='1.0',
      description='nav2py api',
      author='Volodymyr Shcherbyna',
      author_email='dev@voshch.dev',
      packages=find_packages(include=[PROJECT, PROJECT + '.*'])
      )
