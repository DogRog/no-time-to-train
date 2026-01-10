#!/bin/bash
# One-line installation command for no-time-to-train
# Usage: curl -sSL https://raw.githubusercontent.com/.../install.sh | bash
# Or: bash install.sh

set -e

echo "Installing no-time-to-train..."

# Check if in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the no-time-to-train root directory"
    exit 1
fi

# Run the non-interactive setup
chmod +x quick_setup_server.sh
./quick_setup_server.sh

# Download SAM checkpoint
echo "Downloading SAM checkpoint..."
mkdir -p checkpoints
if [ ! -f "checkpoints/sam_vit_h_4b8939.pth" ]; then
    if command -v wget &> /dev/null; then
        wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
    else
        curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o checkpoints/sam_vit_h_4b8939.pth
    fi
else
    echo "SAM checkpoint already exists."
fi

echo ""
echo "Installation complete! Run 'python -c \"import torch; print(torch.cuda.is_available())\"' to verify GPU support."
