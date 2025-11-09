#!/bin/bash
# Quick Setup Script for no-time-to-train
# This script sets up the environment using pip for server deployment

set -e  # Exit on error

echo "========================================"
echo "No-Time-To-Train Quick Setup (pip)"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

required_version="3.10"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10 or higher is required"
    exit 1
fi

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "CUDA version: $cuda_version"
else
    echo "Warning: CUDA not found. PyTorch will be installed with CPU support only."
    echo "For GPU support, please ensure CUDA is installed."
fi

# Create virtual environment (optional but recommended)
echo ""
read -p "Create a virtual environment? (y/n, recommended for clean setup): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    if [ -d "venv" ]; then
        echo "Virtual environment already exists. Skipping creation."
    else
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (adjust CUDA version as needed)
echo ""
echo "Installing PyTorch and torchvision..."
if command -v nvcc &> /dev/null; then
    echo "Installing PyTorch with CUDA 12.1 support..."
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch with CPU support..."
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
fi

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install numpy==2.1.1 \
    pillow==10.4.0 \
    opencv-python==4.10.0.84 \
    tqdm==4.66.5 \
    pyyaml==6.0.2 \
    packaging==24.1 \
    requests==2.32.3

# Install ML/DL frameworks and tools
echo ""
echo "Installing ML/DL frameworks..."
pip install pytorch-lightning==2.1.0 \
    torchmetrics==1.4.3 \
    lightning-utilities==0.11.7 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    jsonargparse==4.33.2

# Install vision and segmentation dependencies
echo ""
echo "Installing computer vision libraries..."
pip install timm==0.4.12 \
    kornia==0.7.4 \
    scikit-image==0.24.0 \
    scikit-learn==1.5.2 \
    scipy==1.14.1 \
    imageio==2.35.1

# Install data processing and analysis tools
echo ""
echo "Installing data processing tools..."
pip install pandas==2.2.3 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    joblib==1.4.2

# Install COCO and annotation tools
echo ""
echo "Installing COCO tools and annotation libraries..."
pip install pycocotools==2.0.8 \
    lvis==0.5.3

# Install fvcore and iopath (for DINOv2/v3)
echo ""
echo "Installing fvcore and iopath..."
pip install fvcore==0.1.5.post20221221 \
    iopath==0.1.10 \
    submitit==1.5.2

# Install experiment tracking and logging
echo ""
echo "Installing experiment tracking tools..."
pip install wandb==0.18.2 \
    tensorboard \
    rich==13.9.2

# Install SAM dependencies
echo ""
echo "Installing Segment Anything dependencies..."
pip install segment-anything==1.0

# Install utilities
echo ""
echo "Installing utility packages..."
pip install addict==2.4.0 \
    termcolor==2.4.0 \
    tabulate==0.9.0 \
    yacs==0.1.8 \
    portalocker==2.10.1 \
    psutil==5.9.8 \
    cython==3.0.11

# Install optimal transport (for matching)
echo ""
echo "Installing optimal transport library..."
pip install pot==0.9.5

# Install mmengine (for some utilities)
echo ""
echo "Installing mmengine..."
pip install mmengine==0.10.5 \
    mmengine-lite==0.10.6

# Install monitoring tools
echo ""
echo "Installing GPU monitoring tools..."
pip install nvitop==1.3.2 \
    nvidia-ml-py==12.535.161

# Install Jupyter (optional for notebooks)
echo ""
read -p "Install Jupyter for notebook support? (y/n): " install_jupyter
if [[ $install_jupyter == "y" || $install_jupyter == "Y" ]]; then
    echo "Installing Jupyter..."
    pip install jupyter==1.1.1 \
        jupyterlab==4.2.5 \
        ipykernel==6.29.5 \
        ipywidgets==8.1.5
fi

# Install the main package in development mode
echo ""
echo "Installing no-time-to-train package..."
pip install -e .

# Install DINOv2
echo ""
echo "Installing DINOv2..."
cd dinov2
pip install -e .
cd ..

# Install DINOv3
echo ""
echo "Installing DINOv3..."
cd dinov3
pip install -e .
cd ..

# Verify installation
echo ""
echo "========================================"
echo "Verifying installation..."
echo "========================================"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torchvision; print(f'torchvision version: {torchvision.__version__}')"
python3 -c "import pytorch_lightning; print(f'PyTorch Lightning version: {pytorch_lightning.__version__}')"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "To activate the environment in the future, run:"
    echo "  source venv/bin/activate"
    echo ""
fi
echo "You can now run experiments using the installed packages."
echo ""
echo "Quick test command:"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
