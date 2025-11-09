# Quick Setup Reference Card

## 🚀 Fast Installation (Recommended for Servers)

```bash
bash quick_setup_server.sh
```

## 📋 Three Setup Options

| Option | Command | Use Case |
|--------|---------|----------|
| **Interactive** | `./quick_setup.sh` | First-time setup, with prompts |
| **Server** | `./quick_setup_server.sh` | Automated/CI-CD deployment |
| **Manual** | See below | Custom configuration |

## 🔧 Manual Setup (4 Steps)

```bash
# 1. Install PyTorch with CUDA
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install main package
pip install -e .

# 4. Install submodules
cd dinov2 && pip install -e . && cd ..
cd dinov3 && pip install -e . && cd ..
```

## 💡 Quick Commands

### Install with different CUDA versions
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Minimal installation
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-minimal.txt
pip install -e .
```

### Verify installation
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📦 Requirements Files

- `requirements.txt` - Full installation with all features
- `requirements-minimal.txt` - Minimal setup without extras

## 🐛 Common Issues

**CUDA extension build fails**: This is normal and can be ignored
```bash
export SAM2_BUILD_CUDA=0
```

**Out of memory**: Install without cache
```bash
pip install --no-cache-dir -r requirements.txt
```

## 📚 More Information

See `SETUP.md` for detailed instructions and troubleshooting.
