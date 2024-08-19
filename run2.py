import cv2
import numpy as np
import sys
import os
import PySimpleGUI as sg
from PIL import Image, ImageTk
from ultralytics import YOLO
sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))
from sort import Sort
import time
from demo import Demo
sg.theme('DarkTeal9')
class main:
    def main():
        demo = Demo()
        
if __name__ == "__main__":
    main.main()

