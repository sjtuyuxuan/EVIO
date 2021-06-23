FROM ros:melodic-ros-base-bionic

USER 0

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y cmake iputils-ping tftp lsb-core wget && \
    apt-get install -y libeigen3-dev && \
    apt-get install -y libopencv-dev && \
    apt-get install -y libgoogle-glog-dev libgflags-dev && \
    apt-get install -y libatlas-base-dev && \
    apt-get install -y libsuitesparse-dev && \
    apt-get install -y zsh && \
    apt-get install -y libpcl-dev && \
    apt-get install -y libceres-dev && \
    apt-get install -y libprotobuf-dev && \
    apt-get clean

RUN apt-get install -y sudo vim && \
    apt-get install -y ros-melodic-pcl-conversions && \
    apt-get install -y ros-melodic-tf\* && \
    apt-get install -y ros-melodic-image-transport-plugins && \
    apt-get install -y openssh-server

RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:borglab/gtsam-release-4.0 && \
    apt update && \
    apt -y install libgtsam-dev libgtsam-unstable-dev

RUN useradd -m yuxuan && echo "yuxuan:yuxuan" | chpasswd && adduser yuxuan sudo

USER yuxuan

RUN cd /home/yuxuan && \
    sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
RUN echo "source /opt/ros/melodic/setup.zsh" >> /home/yuxuan/.zshrc && \
    echo "source /opt/ros/melodic/setup.bash" >> /home/yuxuan/.bashrc && \
    echo "source /develop/devel/setup.zsh" >> /home/yuxuan/.zshrc && \
    echo "source /develop/devel/setup.bash" >> /home/yuxuan/.bashrc

USER 0

RUN echo "deb [arch=amd64 trusted=yes] https://prophesee:DbnLdKL5YXnMndWg@apt.prophesee.ai/dists/public/cp51Vn3b/ubuntu bionic essentials" > /etc/apt/sources.list.d/essentials.list && \
    apt update && \
    apt -y install libcanberra-gtk-module mesa-utils && \
    apt -y install 'metavision-*' && \
    apt -y install libboost-program-options-dev

RUN ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

USER yuxuan



