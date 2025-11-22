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

echo ""
echo "Installation complete! Run 'python -c \"import torch; print(torch.cuda.is_available())\"' to verify GPU support."
