# bitnet-dock

Dockerised [Microsoft BitNet](https://github.com/microsoft/BitNet) inference across a mixed fleet — Intel iMac (CPU) and a PC with an NVIDIA GPU.

Mac mini M4 users: native ARM build is faster and simpler, see [Native ARM instructions](#native-arm-mac-mini-m4).

---

## Fleet map

| Machine | Profile | Path | Notes |
|---|---|---|---|
| Intel iMac | `cpu` | x86 CPU inference | 128 GB RAM, runs large models happily |
| PC (RTX 3090) | `gpu` | CUDA GPU inference | Needs NVIDIA Container Toolkit |
| Mac mini M4 | — | Native ARM build | No Docker needed |

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/bitnet-dock.git
cd bitnet-dock
cp .env.example .env
# edit .env to taste
```

**iMac (CPU):**
```bash
docker compose --profile cpu up --build
```

**PC (GPU):**
```bash
docker compose --profile gpu up --build
```

First run will download the model into a named Docker volume (`bitnet-models` or `bitnet-checkpoints`). Subsequent runs skip the download.

---

## Modes

Set `BITNET_MODE` in `.env`:

| Mode | CPU | GPU | Description |
|---|---|---|---|
| `server` | ✅ | ❌ | OpenAI-compatible HTTP server on `:8080` |
| `chat` | ✅ | ✅ | Interactive terminal chat |
| `benchmark` | ✅ | ✅ | Throughput benchmark |

---

## PC: Prerequisites (resurrect Docker + GPU passthrough)

1. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
2. In Docker Desktop → Settings → Resources → WSL Integration: enable your distro
3. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
4. Verify GPU passthrough:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
   You should see your RTX 3090 listed. If yes, you're ready.

---

## CPU tuning (iMac)

In `.env`, set `N_THREADS` to match your iMac's physical core count.  
For an Intel iMac with 8 cores: `N_THREADS=8`. For 10-core: `N_THREADS=10`.  
Hyperthreads don't help much for inference — stick to physical cores.

---

## Models

The default is `microsoft/BitNet-b1.58-2B-4T` (~0.4 GB weights).

Other supported CPU models (change `BITNET_MODEL` and `BITNET_QUANT` in `.env`):

| HF Repo | Size | Quant |
|---|---|---|
| `1bitLLM/bitnet_b1_58-large` | 0.7B | `i2_s` |
| `HF1BitLLM/Llama3-8B-1.58-100B-tokens` | 8B | `i2_s` |
| `tiiuae/Falcon3-7B-Instruct-1.58bit` | 7B | `i2_s` |
| `tiiuae/Falcon3-10B-Instruct-1.58bit` | 10B | `i2_s` |

The 128 GB iMac can comfortably run the 10B model. The GPU path currently only supports the 2B model.

---

## Native ARM — Mac mini M4

Docker not needed. Native build is faster on Apple Silicon.

```bash
# Prerequisites: Homebrew, conda
brew install llvm cmake
# Make sure clang-18 is on PATH:
export PATH="$(brew --prefix llvm)/bin:$PATH"

git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet

conda create -n bitnet-cpp python=3.11
conda activate bitnet-cpp
pip install -r requirements.txt

huggingface-cli download microsoft/BitNet-b1.58-2B-4T \
    --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# Chat
python run_inference.py \
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p "You are a helpful assistant" -cnv

# Server
python run_inference_server.py \
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    --host 0.0.0.0 --port 8080
```

---

## Persistence

Named Docker volumes mean your downloaded models survive container rebuilds:

```bash
# List volumes
docker volume ls | grep bitnet

# Nuke and redownload (if model corrupted etc.)
docker volume rm bitnet-models
docker volume rm bitnet-checkpoints
```

---

## Repo structure

```
bitnet-dock/
├── Dockerfile.cpu          # iMac / x86 CPU build
├── Dockerfile.gpu          # PC / CUDA build (SM 8.6 = RTX 3090)
├── docker-compose.yml      # profiles: cpu | gpu
├── scripts/
│   ├── entrypoint-cpu.sh   # download + run logic (CPU)
│   └── entrypoint-gpu.sh   # download + convert + run logic (GPU)
├── .env.example            # copy to .env
└── .gitignore
```

---

## Notes

- GPU kernels are compiled at image build time for **SM 8.6** (RTX 3090 / Ampere). Change `TORCH_CUDA_ARCH_LIST` in `Dockerfile.gpu` for other cards.
- The 100B benchmark in Microsoft's README used a **dummy synthetic model** — no 100B ternary model has actually been trained and released. Current largest real model is the 8B Llama3 variant.
- BitNet b1.58 uses `{-1, 0, +1}` weights (ternary), which is distinct from 1-bit binary. The naming is a Microsoft marketing choice.
