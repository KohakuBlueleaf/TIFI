import os
import sys


root_path = os.path.dirname(os.path.realpath(__file__))
resources_path = os.path.join(root_path, "..", "resources")

sys.path.append(os.path.join(resources_path, "MiniGPT-4"))
