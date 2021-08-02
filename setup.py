"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

import setuptools

# with open('VERSION', 'r') as readme_file:
#     version = readme_file.read()

# with open('README.md', 'r') as readme_file:
#     long_description = readme_file.read()

with open('requirements.txt', 'r') as requirements_file:
    requirements = requirements_file.read().splitlines()

setuptools.setup(
    name="simpa_experiments",
    # version=version,
    # url="https://github.com/CAMI-DKFZ/simpa",
    author="Computer Assisted Medical Interventions (CAMI), DKFZ \n"
           "Cancer Research UK, Cambridge Institute (CRUK CI)",
    description="Experiments for the simpa paper",
    # long_description=long_description,
    packages=setuptools.find_packages(),
    # install_requires=requirements
)