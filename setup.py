import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "UKMovementSensing",
    version = "0.0.1",
    description = ("Behaviour detection in wearable movement sensor data"),
    license = "Apache 2.0",
    keywords = "Python",
    url = "https://github.com/NLeSC/UKMovementSensing",
    packages=['UKMovementSensing'],
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
    ],
)
