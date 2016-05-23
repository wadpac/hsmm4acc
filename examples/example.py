#!/usr/bin/env python
""" This example demonstrates that you need to run
'pip install .' in the main directory, before you
can do 'import UKMovementSensing' in a Python program

The from __future__ import below is to make this
code compatible with both Python 2 and 3
"""
from __future__ import print_function

try:
    import UKMovementSensing
    print("Succesfully imported UKMovementSensing!")
except ImportError:
    print("Could not import UKMovementSensing! Maybe you forgot to run 'pip install'")
