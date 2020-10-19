# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution


__version__ = '0.2'
REQUIRED_PACKAGES = []
project_name = 'tensorflow-ctc-ext-beam-search-decoder'


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True


setup(
    name=project_name,
    version=__version__,
    description=('tensorflow-ctc-ext-beam-search-decoder is a custom tensorflow op for ctc beam search that keeps track of the best alignment for each beam'),
    author='Philipp Rouast',
    author_email='philipp@rouast.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    ext_modules=[Extension('_foo', ['stub.cc'])],
    zip_safe=False,
    distclass=BinaryDistribution,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='custom tensorflow op ctc beam search decoder',
)
