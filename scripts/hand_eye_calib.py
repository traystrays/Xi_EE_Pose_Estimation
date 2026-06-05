from pathlib import Path
import pandas as pd
import cv2
import numpy as np

# Using the motion pairs of the end effector and theclassic AX = XB method 
# to find the hand-eye calibration between between ECM and camera

# Using data from Si first, then go to OR to collect data on Xi

ECM_transform = "[-0.00946542528 0.000198544934 0.0764139444 0.160584629 -0.127096385 0.978803575 0.902461052 -0.382696331 -0.197752804 0.399717778 0.915090442 0.0532469451]"