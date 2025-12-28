#!/usr/bin/env python3
"""Quick YOLOv9 model info check"""
import sys
sys.path.insert(0, '/home/orangepi/deepx_sdk/dx_rt/python_package/src')

from dx_engine import InferenceEngine

path = "/home/orangepi/model_for_demo/YOLOv9-S-2.dxnn"
print(f"Loading: {path}")
try:
    ie = InferenceEngine(path)
    print(f"PPU: {ie.is_ppu()}")
    print(f"Input: {ie.get_input_tensors_info()}")
    print(f"Output: {ie.get_output_tensors_info()}")
    print("SUCCESS!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
