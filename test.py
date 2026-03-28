import torch
import cv2
import streamlit
import numpy as np

from src.pipeline import run_pipeline

print("Torch version:", torch.__version__)
print("OpenCV version:", cv2.__version__)
print("Streamlit version:", streamlit.__version__)

dummy = np.zeros((32, 32, 3), dtype=np.uint8)
out = run_pipeline(dummy)
print("Pipeline predicted noise:", out.predicted_noise)
print("Environment OK")
