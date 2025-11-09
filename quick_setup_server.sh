#!/bin/bash
# Non-interactive Quick Setup Script for no-time-to-train
# Use this for automated server deployment

set -e  # Exit on error

echo "========================================"
echo "No-Time-To-Train Quick Setup (Non-interactive)"
echo "========================================"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies from requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install the main package
echo "Installing no-time-to-train package..."
pip install -e .

# Install DINOv2
echo "Installing DINOv2..."
cd dinov2 && pip install -e . && cd ..

# Install DINOv3
echo "Installing DINOv3..."
cd dinov3 && pip install -e . && cd ..

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "Setup Complete!"
