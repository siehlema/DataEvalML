from setuptools import setup

setup(
    name='senvaleva',
    py_modules=['senvaleva'],
    version='0.1',
    install_requires=[
        'matplotlib',
        'numpy',
        'tensorflow>=1.8.0',
        'keras',
        'sklearn',
        'tpot'
    ],
    description="Repository for Sensor Value Evaluation using Machine Learning",
    author="Martin Siehler",
)
