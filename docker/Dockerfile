FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

WORKDIR /fatigue_driver

# disable terminal interaction for apt
ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list
COPY sources.list /etc/apt/sources.list
# # Python basics
RUN apt-get update && apt-get install -y \
        python3-opencv \
        && rm -rf /var/lib/apt/lists/*


COPY requirement.txt /fatigue_driver/requirement.txt

RUN pip3 install -r requirement.txt -i https://pypi.doubanio.com/simple



#  CUDA和pytorch的对应关系
# CUDA 11.6 - pytorch 1.12.0
# conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge

# Docker从该仓库进行参考
# https://github.com/slabban/ros2_pytorch_cuda

# RUN rm /etc/apt/sources.list
# COPY sources.list /etc/apt/sources.list

# # disable terminal interaction for apt
# ENV DEBIAN_FRONTEND=noninteractive

# # # Install cudnn8 and move necessary header files to cuda include directory
# RUN apt-get update && apt install libcudnn8-dev -y && \
# 	cp /usr/include/cudnn_version.h /usr/local/cuda/include && \
# 	cp /usr/include/cudnn.h /usr/local/cuda/include/ && \
# 	rm -rf /var/lib/apt/lists/*


# # Fundamentals
# RUN apt-get update && apt-get install -y \
#         bash-completion \
#         build-essential \
#         clang-format \
#         cmake \
#         curl \
#         git \
#         gnupg2 \
#         locales \
#         lsb-release \
#         rsync \
#         software-properties-common \
#         wget \
#         vim \
#         unzip \
#         mlocate \
# 	    libgoogle-glog-dev \
#         && rm -rf /var/lib/apt/lists/*


# # # Python basics
# RUN apt-get update && apt-get install -y \
#         python3-flake8 \
#         python3-opencv \
#         python3-pip \
#         python3-pytest-cov \
#         python3-setuptools \
#         && rm -rf /var/lib/apt/lists/*

# # # Python3 (PIP)
# RUN python3 -m pip install -U \
#         argcomplete \
#         autopep8 \
#         flake8 \
#         flake8-blind-except \
#         flake8-builtins \
#         flake8-class-newline \
#         flake8-comprehensions \
#         flake8-deprecated \
#         flake8-docstrings \
#         flake8-import-order \
#         flake8-quotes \
#         onnx \
#         pytest-repeat \
#         pytest-rerunfailures \
#         pytest \
#         pydocstyle \
#         scikit-learn \
#         torch \
#         torchvision \
#         -i https://pypi.doubanio.com/simple

# # Install libtorch
# # RUN wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcu116.zip && \
# #         unzip libtorch-cxx11-abi-shared-with-deps-1.12.0+cu116.zip && \
# #         rm -rf libtorch-cxx11-abi-shared-with-deps-1.12.0+cu116.zip 
        
# # Setup ROS2 Foxy
# RUN locale-gen en_US en_US.UTF-8
# RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
# ENV LANG=en_US.UTF-8

# # In case of website not accessible, use downloaded file
# # RUN curl -ssl https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
# # COPY ros.key /usr/share/keyrings/ros-archive-keyring.gpg
# RUN curl http://repo.ros2.org/repos.key | apt-key add - 
# RUN sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list'

# RUN apt-get update && apt-get install -y \
#         python3-colcon-common-extensions \
#         python3-rosdep \
#         python3-vcstool \
#         python3-argcomplete \
#         && rm -rf /var/lib/apt/lists/*


# RUN sh -c 'echo "deb [arch=amd64] http://mirrors.aliyun.com/ros2/ubuntu/ focal main" > /etc/apt/sources.list.d/ros2-latest.list'      
# RUN apt-get update && apt-get install ros-foxy-desktop -y \
#         && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install ros-foxy-rmw-cyclonedds-cpp -y 

# RUN rosdep init

# RUN rosdep update