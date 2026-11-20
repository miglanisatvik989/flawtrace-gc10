#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /home/soft/anaconda3/etc/profile.d/conda.sh
if ! conda activate flawtrace-gc10 2>/dev/null; then
    conda create --yes --name flawtrace-gc10 --override-channels -c conda-forge \
        python=3.12.14 uv=0.12.5 libarchive=3.8.11
    conda activate flawtrace-gc10
fi

# The OpenCV wheels share cv2 files. Reinstall headless after removing a conflict.
if uv pip list --python "$CONDA_PREFIX/bin/python" --format freeze | grep -E '^opencv-(python|contrib-python(-headless)?)==' >/dev/null; then
    uv pip uninstall --python "$CONDA_PREFIX/bin/python" \
        opencv-python opencv-contrib-python opencv-contrib-python-headless opencv-python-headless
fi

uv pip install --python "$CONDA_PREFIX/bin/python" \
    --index https://download.pytorch.org/whl/cu128 \
    --default-index https://pypi.org/simple \
    --index-strategy unsafe-first-match \
    --requirements <(sed 's/^opencv-python==/opencv-python-headless==/' requirements.txt)
uv pip check --python "$CONDA_PREFIX/bin/python"
"$CONDA_PREFIX/bin/python" -c 'import cv2; print("OpenCV", cv2.__version__)'
bsdtar --version
