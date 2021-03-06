""" Setup for midaspy
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='midaspy',

    version='0.0.1',

    description='Python module for mixed frequency data sampling (MIDAS) regression modeling',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/yoseph-zuskin/midaspy',

    # Author details
    author='Yoseph Zuskin',
    author_email='zuskinyoseph@gmail.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='regression midas timeseries mixedfrequency forecasting economics econometric',

    packages=find_packages(exclude=['docs', 'tests*']),

    install_requires=['pandas', 'numpy', 'scipy', 'python-dateutil'],

)