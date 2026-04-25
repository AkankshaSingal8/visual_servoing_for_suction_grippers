#!/usr/bin/env bash
# SemVS environment + third-party install for negative_weighing.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-semvs}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
# cuda: install torch+torchvision from PyTorch cu124 wheels. cpu: pip defaults.
SEMVS_TORCH="${SEMVS_TORCH:-cuda}"
# Set DOWNLOAD_WEIGHTS=0 to skip checkpoint downloads (large).
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
# pip install stereo SDK bindings (only works if ZED SDK is already on the system).
INSTALL_PYZED="${INSTALL_PYZED:-0}"
# Robot control (XArm) used when not in camera-only / test paths.
INSTALL_XARM="${INSTALL_XARM:-1}"

die() {
  echo "error: $*" >&2
  exit 1
}

command -v conda >/dev/null 2>&1 || die "conda not found; install Miniconda/Anaconda first."

for d in GroundingDINO sam2 Depth-Anything-V2; do
  [[ -d "${SCRIPT_DIR}/third-party/${d}" ]] || die "missing ${SCRIPT_DIR}/third-party/${d}. Clone with: git clone --recurse-submodules <SemVS-url> && cd SemVS"
done

have_env() {
  conda run -n "${CONDA_ENV_NAME}" true 2>/dev/null
}

if ! have_env; then
  echo "Creating conda env: ${CONDA_ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
else
  echo "Reusing existing conda env: ${CONDA_ENV_NAME}"
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

echo "Installing PyTorch (${SEMVS_TORCH})..."
if [[ "${SEMVS_TORCH}" == "cpu" ]]; then
  pip install --upgrade pip
  pip install "torch>=2.5.1" "torchvision>=0.20.1"
else
  pip install --upgrade pip
  pip install "torch>=2.5.1" "torchvision>=0.20.1" \
    --index-url https://download.pytorch.org/whl/cu124
fi

echo "Installing shared Python dependencies..."
pip install \
  opencv-python \
  numpy \
  pillow \
  tqdm \
  scipy \
  "hydra-core>=1.3.2" \
  "omegaconf>=2.3.0" \
  "iopath>=0.1.10" \
  transformers \
  supervision \
  pycocotools \
  timm \
  addict \
  yapf \
  ninja

echo "Installing GroundingDINO (editable, builds CUDA/C++ ops when GPU toolkit is available)..."
pip install -e "${SCRIPT_DIR}/third-party/GroundingDINO"

echo "Installing SAM 2 (editable)..."
pip install -e "${SCRIPT_DIR}/third-party/sam2"

echo "Depth-Anything-V2 has no setuptools package; adding repo root to PYTHONPATH for this env..."
ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"
cat > "${ACTIVATE_DIR}/semvs_pythonpath.sh" <<EOF
_SEMVS_DA_ROOT="${SCRIPT_DIR}/third-party/Depth-Anything-V2"
if [[ "\${PYTHONPATH:-}" != *"\${_SEMVS_DA_ROOT}"* ]]; then
  export _SEMVS_PYTHONPATH_BACKUP="\${PYTHONPATH:-}"
  export PYTHONPATH="\${_SEMVS_DA_ROOT}\${PYTHONPATH:+:\${PYTHONPATH}}"
fi
unset _SEMVS_DA_ROOT
EOF
cat > "${DEACTIVATE_DIR}/semvs_pythonpath.sh" <<EOF
if [[ -n "\${_SEMVS_PYTHONPATH_BACKUP+x}" ]]; then
  if [[ -n "\${_SEMVS_PYTHONPATH_BACKUP}" ]]; then
    export PYTHONPATH="\${_SEMVS_PYTHONPATH_BACKUP}"
  else
    unset PYTHONPATH
  fi
  unset _SEMVS_PYTHONPATH_BACKUP
fi
EOF
# Apply for current shell session
export PYTHONPATH="${SCRIPT_DIR}/third-party/Depth-Anything-V2${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${INSTALL_PYZED}" == "1" ]]; then
  echo "Installing pyzed (requires Stereolabs ZED SDK installed on the system)..."
  pip install pyzed || die "pyzed install failed; install ZED SDK from https://www.stereolabs.com/developers/release/"
fi

if [[ "${INSTALL_XARM}" == "1" ]]; then
  echo "Installing xArm Python SDK..."
  pip install xarm-python-sdk
fi

download_file() {
  local url="$1"
  local dest="$2"
  if [[ -f "${dest}" ]]; then
    echo "  (skip) already present: ${dest}"
    return 0
  fi
  echo "  downloading -> ${dest}"
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "${dest}" "${url}"
  else
    curl -fL --progress-bar -o "${dest}" "${url}"
  fi
}

if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
  echo "Downloading model checkpoints (set DOWNLOAD_WEIGHTS=0 to skip)..."
  mkdir -p "${SCRIPT_DIR}/third-party/GroundingDINO/weights"
  mkdir -p "${SCRIPT_DIR}/third-party/sam2/checkpoints"
  mkdir -p "${SCRIPT_DIR}/third-party/Depth-Anything-V2/checkpoints"

  download_file \
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
    "${SCRIPT_DIR}/third-party/GroundingDINO/weights/groundingdino_swint_ogc.pth"

  # negative_weighing.py uses sam2.1_hiera_l + SAM2_CKPT sam2.1_hiera_large.pt
  download_file \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
    "${SCRIPT_DIR}/third-party/sam2/checkpoints/sam2.1_hiera_large.pt"

  # DA_ENCODER defaults to vits -> depth_anything_v2_vits.pth
  download_file \
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth" \
    "${SCRIPT_DIR}/third-party/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
else
  echo "Skipping checkpoint downloads (DOWNLOAD_WEIGHTS=0)."
fi

cat <<EOF

Done.

Activate later:
  conda activate ${CONDA_ENV_NAME}

Run:
  cd ${SCRIPT_DIR}
  python negative_weighing.py

Notes:
  - Depth-Anything-V2 is on PYTHONPATH via ${CONDA_PREFIX}/etc/conda/activate.d/semvs_pythonpath.sh
  - ZED camera: install the ZED SDK, then re-run with INSTALL_PYZED=1 ./setup.sh (or pip install pyzed in the env)
  - For CPU-only PyTorch: SEMVS_TORCH=cpu ./setup.sh
  - GroundingDINO needs a working CUDA toolkit + compiler for GPU ops when SEMVS_TORCH=cuda; for CPU-only stack use SEMVS_TORCH=cpu (may still need build tools for the extension)
EOF
