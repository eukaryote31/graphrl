from setuptools import setup, find_packages


setup(
    name='graphrl',
    version='0.0.1',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    setup_requires=[
        'pytest-runner'
    ],
    install_requires=[
        'torch==0.4.1',
        'numpy',
        'networkx'
    ],
    tests_require=[
        'pytest'
    ]
)
