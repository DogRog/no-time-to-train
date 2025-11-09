# Quick Setup Guide

This guide provides instructions for setting up the `no-time-to-train` project using pip on a server environment.

## Prerequisites

- Python 3.10 or higher
- CUDA 12.1+ (for GPU support)
- pip package manager

## Setup Options

### Option 1: Interactive Setup (Recommended for first-time setup)

This script will prompt you for choices and create a virtual environment:

```bash
chmod +x quick_setup.sh
./quick_setup.sh
```

Features:
- Prompts for virtual environment creation
- Prompts for Jupyter installation
- Verifies installations
- Provides detailed progress information

### Option 2: Non-Interactive Setup (For automated deployment)

This script runs without prompts, suitable for CI/CD or remote deployment:

```bash
chmod +x quick_setup_server.sh
./quick_setup_server.sh
```

Features:
- No user interaction required
- Installs all dependencies automatically
- Uses CUDA 12.1 by default

### Option 3: Manual Installation

For custom setups or troubleshooting:

```bash
# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install the package and submodules
pip install -e .
cd dinov2 && pip install -e . && cd ..
cd dinov3 && pip install -e . && cd ..
```

## CUDA Version Configuration

If you have a different CUDA version, modify the PyTorch installation command:

- **CUDA 11.8**: 
  ```bash
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
  ```

- **CUDA 12.4**:
  ```bash
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
  ```

- **CPU Only**:
  ```bash
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
  ```

## Verification

After installation, verify the setup:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import pytorch_lightning; print(f'Lightning version: {pytorch_lightning.__version__}')"
```

## Optional Components

### Without Jupyter

If you don't need Jupyter notebooks, comment out the Jupyter section in `requirements.txt`:

```
# Jupyter (optional, comment out if not needed)
# jupyter==1.1.1
# jupyterlab==4.2.5
# ipykernel==6.29.5
# ipywidgets==8.1.5
```

### Minimal Installation

For a minimal setup without monitoring tools and Jupyter:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==2.1.0 hydra-core==1.3.2 omegaconf==2.3.0
pip install pycocotools==2.0.8 opencv-python==4.10.0.84
pip install -e .
```

## Troubleshooting

### CUDA Extension Build Errors

If you encounter errors building CUDA extensions for SAM2, this is expected and won't affect most functionality. The setup script is configured to continue despite these errors.

To disable CUDA extension building:
```bash
export SAM2_BUILD_CUDA=0
pip install -e .
```

### Memory Issues

If pip runs out of memory during installation:
```bash
pip install --no-cache-dir -r requirements.txt
```

### Virtual Environment

To activate the virtual environment after creation:
```bash
source venv/bin/activate
```

To deactivate:
```bash
deactivate
```

## Server-Specific Notes

- The scripts assume a GPU is available with CUDA 12.1
- For headless servers, X11 forwarding is not required
- All matplotlib backends will work in non-interactive mode
- WandB logging works without browser interaction

## Next Steps

After successful installation:

1. Test the installation with a simple experiment
2. Configure your dataset paths in the config files
3. Run the example scripts in the `scripts/` directory

For more information, see the main README.md file.
