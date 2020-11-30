import os
import sys

print("simulator_testGround.py is called. __file__:", __file__)
print("simulator_testGround.py __name__:", __name__)
print("simulator_testGround.py __package__:", __package__)
print("simulator_testGround.py os.path.abspath(__file__):", os.path.abspath(__file__))
print("simulator_testGround.py os.path.dirname(os.path.abspath(__file__)):", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Utils.loaded_test import *

print_loaded_test_info()
