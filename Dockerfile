# NOTE: This is a docker file for running our code with CuOpt. The pip wheels must be built first and copied to the built_wheels directory.

# run by: sudo docker run -it --rm --runtime=nvidia --gpus all  nvcr.io/nvidian/nvr-rock/rloptimizer:latest /bin/bash
# push by: sudo docker push nvcr.io/nvidian/nvr-rock/rloptimizer
# build by:  sudo docker build --network=host -t nvcr.io/nvidian/nvr-rock/rloptimizer:1.10 -f Dockerfile .

# Build Wheels
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
ENV DEBIAN_FRONTEND=nonintercative

RUN apt-get update &&  apt-get install -y \
    bc vim mc git git-lfs\
    software-properties-common \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update &&  apt-get install -y \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'alias python="python3"' >> ~/.bashrcexit

RUN pip install cuopt-server-cu12==25.5.* cuopt-sh-client==25.5.* nvidia-cuda-runtime-cu12==12.8.* pyyaml scikit-build-core einops tensordict  gymnasium stable-baselines3[extra]  colorama  ruamel.yaml networkx numpy scipy wandb jupyterlab tensorboard torchrl seaborn pyvrp pyyaml matplotlib opencv-python-headless geopandas contextily tqdm --extra-index-url=https://pypi.nvidia.com && pip cache purge

RUN git clone https://gitlab.com/igreenberg/initializable_hgs /tmp/hygese
WORKDIR /tmp/hygese/
RUN mkdir -p lib/build lib/bin && python setup.py build_py && pip install -e .

WORKDIR /workspace
SHELL ["/bin/bash", "-c"]
