#!/usr/bin/env python3

#----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='sstudy',
      version='0.0.3',
      description='Simulation study storage',
      author='Marco Inacio',
      author_email='pythonpackages@marcoinacio.com',
      packages=['sstudy'],
      keywords = ['simulation study', ],
      license='GPL3',
      install_requires=['peewee']
     )
