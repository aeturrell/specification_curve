#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['statsmodels', 'matplotlib', 'numpy', 'pandas',
                'scikit-learn']

setup_requirements = [ ]

test_requirements = ['numpy', 'pandas']

setup(
    author="Arthur Turrell",
    author_email='a.turrell09@imperial.ac.uk',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Specification Curve is a Python package that performs specification curve analysis.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='specification_curve',
    name='specification_curve',
    packages=find_packages(include=['specification_curve', 'specification_curve.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/aeturrell/specification_curve',
    version='0.2.2',
    zip_safe=False,
)
