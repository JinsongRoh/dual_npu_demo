#!/bin/bash
# Production Dual NPU App Launcher
# Clean solution: uses opencv-python-headless (no Qt bundled)

set -e

cd /home/orangepi/dual_npu_demo

# Auto-detect display if not set
if [ -z "$DISPLAY" ]; then
    # Check if running in graphical session
    if [ -n "$XDG_SESSION_TYPE" ] && [ "$XDG_SESSION_TYPE" = "x11" ]; then
        export DISPLAY=:0
    elif [ -e "/tmp/.X11-unix/X0" ]; then
        export DISPLAY=:0.0
    else
        echo "Error: No display available. Run with: DISPLAY=:0 $0"
        exit 1
    fi
fi

# Qt plugin paths for PyQt5 (system Qt, not opencv's bundled Qt)
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms
export QT_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins

# Verify critical files exist
if [ ! -f "production_app.py" ]; then
    echo "Error: production_app.py not found in $(pwd)"
    exit 1
fi

if [ ! -f "/home/orangepi/model_for_demo/YOLOv5s_640.dxnn" ]; then
    echo "Error: Model file not found"
    exit 1
fi

# Run
exec python3 production_app.py "$@"
