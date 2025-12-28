#!/usr/bin/env python3
"""Debug YOLOv9 model loading"""
import sys
import os
import traceback

sys.path.insert(0, '/home/orangepi/deepx_sdk/dx_rt/python_package/src')

path = "/home/orangepi/model_for_demo/YOLOv9-S-2.dxnn"

# Check file exists
print(f"File exists: {os.path.exists(path)}")
print(f"File size: {os.path.getsize(path) if os.path.exists(path) else 'N/A'}")

try:
    print("Importing dx_engine...")
    from dx_engine import InferenceEngine
    print("Import successful!")

    print(f"Loading model: {path}")
    sys.stdout.flush()
    sys.stderr.flush()

    ie = InferenceEngine(path)

    print(f"PPU: {ie.is_ppu()}")
    print(f"Input: {ie.get_input_tensors_info()}")
    print(f"Output: {ie.get_output_tensors_info()}")
    print("SUCCESS!")

except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
