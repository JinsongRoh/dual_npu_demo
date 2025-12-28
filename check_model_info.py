#!/usr/bin/env python3
"""Quick model info check"""
import sys
sys.path.insert(0, '/home/orangepi/deepx_sdk/dx_rt/python_package/src')

from dx_engine import InferenceEngine

models = [
    ("/home/orangepi/model_for_demo/YOLOv5s_640.dxnn", "YOLOv5s"),
    ("/home/orangepi/model_for_demo/YOLOv9-S-2.dxnn", "YOLOv9-S"),
]

for path, name in models:
    print(f"\n{'='*50}")
    print(f"{name}: {path}")
    print('='*50)
    try:
        ie = InferenceEngine(path)
        print(f"  PPU: {ie.is_ppu()}")
        print(f"  Input: {ie.get_input_tensors_info()}")
        print(f"  Output: {ie.get_output_tensors_info()}")
    except Exception as e:
        print(f"  Error: {e}")
