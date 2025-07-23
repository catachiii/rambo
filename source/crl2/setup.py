#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

setup(
    name="crl2",
    version="0.0.1",
    packages=find_packages(),
    author="Computational Robotics Lab, ETH Zurich",
    maintainer="Jin Cheng",
    maintainer_email="jicheng@ethz.ch",
    url="git@gitlab.inf.ethz.ch:jicheng/crl2.git",
    license="BSD-3",
    description="Fast and simple RL algorithms implemented in pytorch",
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
        "GitPython",
        "onnx",
        "moviepy",
        "imageio",
    ],
)
