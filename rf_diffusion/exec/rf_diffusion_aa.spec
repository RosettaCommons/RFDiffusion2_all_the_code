Bootstrap: docker
From: nvidia/cuda@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
IncludeCmd: yes

%labels
    BaseImage nvidia/cuda:12.8.1-devel-ubuntu24.04
    BaseDigest sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
    CUDA 12.8.1
    Torch 2.8.0+cu128
    DGLCommit ba731332bd4103b43a2aa0e6b4ea8c675b57bc19
    RoseTTAFold2Commit 4b273b95ba05ce7524d225bfe1711b8fa76a6a11
    CUTLASSCommit f7b19de32c5d1f3cedfc735c2849f12b537522ee
    Miniforge 26.3.2-3

%post
set -eu

export DEBIAN_FRONTEND=noninteractive
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"
export CMAKE_CUDA_ARCHITECTURES="80;86;89;90;120"
export CUDA_ARCHITECTURES="80;86;89;90;120"
export CUDAARCHS="80;86;89;90;120"

apt-get update
apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    bzip2 \
    ca-certificates \
    cmake \
    curl \
    g++-11 \
    gcc-11 \
    git \
    libaio-dev \
    libcurl4-openssl-dev \
    libjpeg-dev \
    libpng-dev \
    libssl-dev \
    libx11-6 \
    libxau6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    pkg-config \
    rsync \
    software-properties-common \
    wget
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 50
update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-11 50

add-apt-repository -y ppa:apptainer/ppa
apt-get update
apt-get install -y --no-install-recommends apptainer

# Runtime bind points only. Databases, checkpoints, repositories, and user
# files remain external to the image.
mkdir -p /databases /software /home /projects /net /mnt/home /mnt/projects /mnt/net /mnt/databases /mnt/software

MINIFORGE_VERSION=26.3.2-3
MINIFORGE_SHA256=848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"
curl -fsSL "${MINIFORGE_URL}" -o /opt/miniforge.sh
echo "${MINIFORGE_SHA256}  /opt/miniforge.sh" | sha256sum -c -
bash /opt/miniforge.sh -b -u -p /opt/conda
export PATH=/opt/rf2aa/bin:/usr/local/cuda/bin:${PATH}
export CONDA_PREFIX=/opt/rf2aa

/opt/conda/bin/conda create --yes -p /opt/rf2aa \
    -c https://conda.rosettacommons.org \
    -c conda-forge \
    python=3.12 \
    pip \
    "numpy<2" \
    zlib=1.2.13 \
    libzlib=1.2.13 \
    openbabel=3.1.1=py312h8a8b3d1_8 \
    pyrosetta=2025.03+release.1f5080a079=py312_0

/opt/conda/bin/conda clean -a -y
rm -rf /opt/conda/pkgs/*

python -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging
python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    'torch==2.8.0+cu128'
printf '%s\n' \
    'numpy<2' \
    'sympy==1.14.0' \
    'torch==2.8.0+cu128' \
    > /opt/rf2aa-pip-constraints.txt
export PIP_CONSTRAINT=/opt/rf2aa-pip-constraints.txt
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128

# Build DGL from a pinned public source checkout because official DGL wheels do
# not cover the Torch/CUDA pair needed for Blackwell.
python -m pip install --no-cache-dir "cython<3" networkx "numpy<2" psutil "pydantic>=2" pyyaml requests scipy==1.13.1 tqdm
git clone --recursive https://github.com/dmlc/dgl.git /opt/dgl
cd /opt/dgl
git checkout --detach ba731332bd4103b43a2aa0e6b4ea8c675b57bc19
git submodule sync --recursive
git submodule update --init --recursive
# DGL 2.5.x uses an unanchored GraphBolt arch filter that drops Blackwell
# "120" because it contains the substring "20".
sed -i 's/EXCLUDE REGEX "\[2-6\]\[0-9\]"/EXCLUDE REGEX "^[2-6][0-9]$"/' graphbolt/CMakeLists.txt
export DGL_HOME=/opt/dgl
mkdir -p build
cd build
cmake \
    -DBUILD_TYPE=release \
    -DUSE_CUDA=ON \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN=80\;86\;89\;90\;120 \
    -DCUDA_ARCH_PTX=120 \
    -DCMAKE_CUDA_ARCHITECTURES=80\;86\;89\;90\;120 \
    -DCUDA_ARCHITECTURES=80\;86\;89\;90\;120 \
    -DTORCH_PYTHON_INTERPS="$(command -v python)" \
    ..
DGL_BUILD_JOBS="$(nproc)"
if [ "${DGL_BUILD_JOBS}" -gt 16 ]; then
    DGL_BUILD_JOBS=16
fi
cmake --build . --parallel "${DGL_BUILD_JOBS}"
cd /opt/dgl/python
python setup.py install
python setup.py build_ext --inplace
cd /
rm -rf /opt/dgl

python -m pip install --no-cache-dir \
    hydra-core==1.3.1 \
    ml-collections==0.1.1 \
    addict==2.4.0 \
    assertpy==1.1.0 \
    biopython==1.83 \
    colorlog \
    compact-json \
    cython==3.0.0 \
    cytoolz==0.12.3 \
    debugpy==1.8.5 \
    deepdiff==6.3.0 \
    dm-tree==0.1.8 \
    e3nn==0.5.1 \
    einops==0.7.0 \
    executing==2.0.0 \
    fastparquet==2024.5.0 \
    fire==0.6.0 \
    GPUtil==1.4.0 \
    icecream==2.1.3 \
    ipdb==0.13.11 \
    ipykernel==6.29.5 \
    ipython==8.27.0 \
    ipywidgets \
    mdtraj==1.10.0 \
    numba \
    omegaconf==2.3.0 \
    opt_einsum==3.3.0 \
    pandas==2.2.3 \
    plotly==5.16.1 \
    pre-commit==3.7.1 \
    py3Dmol==2.2.1 \
    pyarrow==17.0.0 \
    pydantic \
    pyrsistent==0.19.3 \
    pytest-benchmark \
    pytest-cov==4.1.0 \
    pytest-dotenv==0.5.2 \
    pytest==8.2.0 \
    rdkit==2024.3.5 \
    RestrictedPython \
    ruff==0.6.2 \
    scipy==1.13.1 \
    seaborn==0.13.2 \
    sympy==1.14.0 \
    tmtools \
    tqdm==4.65.0 \
    typer==0.12.5 \
    wandb==0.16.6

# Biotite fork used by RFdiffusion-AA.
ln -sf /usr/include/*.h /usr/local/include/
env -u PIP_CONSTRAINT python -m pip install --no-cache-dir --no-deps \
    git+https://github.com/biotite-dev/biotite.git@fab175e7ba4608d9613f092ad4e080661c6cc816
python -m pip install --no-cache-dir biotraj==1.2.2 msgpack==1.1.2

# PyG CUDA extension wheels matched to Torch 2.8 / CUDA 12.8.
python -m pip install --no-cache-dir --no-deps \
    'pyg_lib==0.6.0+pt28cu128' \
    'torch_scatter==2.1.2+pt28cu128' \
    'torch_sparse==0.6.18+pt28cu128' \
    'torch_cluster==1.6.3+pt28cu128' \
    'torch_spline_conv==1.2.2+pt28cu128' \
    -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# Runtime dependency for cuequivariant.
python -m pip install --no-cache-dir \
    --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    pylibcugraphops-cu12==24.12.0

python -m pip install --no-cache-dir deepspeed==0.15.1

python -m pip install --no-cache-dir git+https://github.com/RalphMao/PyTimer.git@c81aa706afe14bb4fb60cdd0f9137350fac5853d
python -m pip install --no-cache-dir git+https://github.com/baker-laboratory/ipd.git@3959d1fd5acce6a4501b9c034a40a2a930f21bf2
python -m pip uninstall -y ipd

python -m biotite.setup_ccd

git clone --branch v3.5.1 --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass
cd /opt/cutlass
git checkout --detach f7b19de32c5d1f3cedfc735c2849f12b537522ee

git clone https://github.com/uw-ipd/RoseTTAFold2.git /opt/RoseTTAFold2
cd /opt/RoseTTAFold2
git checkout --detach 4b273b95ba05ce7524d225bfe1711b8fa76a6a11
git submodule update --init --recursive
python -m pip install --no-cache-dir pynvml==11.0.0 git+https://github.com/NVIDIA/dllogger.git@0478734ff7be75adde8d160e04872664d1c62e5f
python -m pip install --no-cache-dir --no-deps /opt/RoseTTAFold2/SE3Transformer

/opt/conda/bin/conda clean -a -y
python -m pip cache purge || true
apt-get -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/* /opt/miniforge.sh

%environment
export PATH=/opt/rf2aa/bin:/usr/local/cuda/bin:$PATH
export CONDA_PREFIX=/opt/rf2aa
export CUTLASS_PATH=/opt/cutlass
export MKL_SERVICE_FORCE_INTEL=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"
export CMAKE_CUDA_ARCHITECTURES="80;86;89;90;120"
export CUDA_ARCHITECTURES="80;86;89;90;120"
export CUDAARCHS="80;86;89;90;120"
export DGL_LIBRARY_PATH=/opt/rf2aa/dgl

%runscript
exec python "$@"

%help
RFdiffusion-AA runtime image with CUDA 12.8.1, PyTorch 2.8.0+cu128, DGL built from the pinned upstream 2.5.x branch, and Blackwell-capable extension build flags.

This image intentionally does not embed model checkpoints, databases, RFdiffusion-AA source trees, or lab filesystem symlinks. Bind those resources at runtime, for example:

  apptainer exec --nv \
    --bind /path/to/rf_diffusion_aa:/opt/rf_diffusion_aa \
    --bind /path/to/models:/models \
    --bind /path/to/databases:/databases \
    rf_diffusion_aa.sif python /opt/rf_diffusion_aa/rf_diffusion/run_inference.py ...

Important source pins:

  Base image: nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
  PyTorch: 2.8.0+cu128
  DGL: ba731332bd4103b43a2aa0e6b4ea8c675b57bc19
  RoseTTAFold2: 4b273b95ba05ce7524d225bfe1711b8fa76a6a11
  CUTLASS: v3.5.1 / f7b19de32c5d1f3cedfc735c2849f12b537522ee
  Miniforge: 26.3.2-3 / sha256:848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c
