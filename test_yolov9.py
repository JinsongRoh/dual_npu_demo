#!/usr/bin/env python3
"""
Test YOLOv9-S model output format on DX-M1 NPU
Compare with YOLOv5s to verify PPU compatibility
"""
import sys
sys.path.insert(0, '/home/orangepi/deepx_sdk/dx_rt/python_package/src')

import numpy as np
import struct
from dx_engine import InferenceEngine

def test_model(model_path, model_name):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print('='*60)

    try:
        ie = InferenceEngine(model_path)

        # Get model info
        input_info = ie.get_input_tensors_info()
        output_info = ie.get_output_tensors_info()
        is_ppu = ie.is_ppu()

        print(f"\nModel Info:")
        print(f"  Is PPU: {is_ppu}")
        print(f"  Input tensors: {input_info}")
        print(f"  Output tensors: {output_info}")

        # Create dummy input (640x640x3)
        dummy_input = np.random.randint(0, 255, (1, 640, 1920), dtype=np.uint8)

        # Run inference
        outputs = ie.run([dummy_input.flatten()])

        print(f"\nOutput Info:")
        print(f"  Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"  Output {i}:")
            print(f"    Shape: {out.shape}")
            print(f"    Dtype: {out.dtype}")
            print(f"    Size (bytes): {out.nbytes}")

            if out.nbytes > 0:
                # Check if it's DeviceBoundingBox_t format (32 bytes per detection)
                if out.nbytes % 32 == 0:
                    num_dets = out.nbytes // 32
                    print(f"    Possible detections (32-byte format): {num_dets}")

                    if num_dets > 0 and num_dets < 100:
                        # Try parsing first few detections
                        raw = out.tobytes()
                        print(f"\n    First detection parse attempt:")
                        for j in range(min(3, num_dets)):
                            try:
                                data = struct.unpack('<4f4BfI4x', raw[j*32:(j+1)*32])
                                cx, cy, w, h = data[0], data[1], data[2], data[3]
                                grid_y, grid_x, box_idx, layer_idx = data[4], data[5], data[6], data[7]
                                score = data[8]
                                label = data[9]

                                print(f"    Det {j}: cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}")
                                print(f"            score={score:.3f}, label={label}")
                            except Exception as e:
                                print(f"    Det {j}: Parse error - {e}")

        print(f"\n✓ Model loaded and inference successful!")
        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    models = [
        ("/home/orangepi/model_for_demo/YOLOv5s_640.dxnn", "YOLOv5s (640)"),
        ("/home/orangepi/model_for_demo/YOLOv9-S-2.dxnn", "YOLOv9-S (640)"),
    ]

    results = {}
    for path, name in models:
        results[name] = test_model(path, name)

    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    for name, success in results.items():
        status = "✓ OK" if success else "✗ FAILED"
        print(f"  {name}: {status}")
