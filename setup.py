#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2020-2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""setup package."""
from setuptools import find_packages, setup

setup(
    name="mindspore_rl",
    version="0.8.0",
    author="The MindSpore Authors",
    author_email="contact@mindspore.cn",
    description="A MindSpore reinforcement learning framework.",
    url="https://github.com/mindspore-lab/mindrl",
    packages=find_packages(include=["mindspore_rl*"]),
    package_data={
        "mindspore_rl": ["distribution/distribution_policies/*/*.tp"],
    },
    install_requires=[
        "numpy>=1.17.0",
        "matplotlib>=3.1.3",
        "gym>=0.18.3",
        "pyyaml == 6.0",
        "astor == 0.8.1",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    license="Apache 2.0",
    keywords="mindspore reinforcement learning",
)
