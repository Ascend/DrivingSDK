FROM quay.io/pypa/manylinux_2_28_aarch64:latest

RUN cd /usr/local/bin \
    && ln -s /opt/_internal/cpython-3.7.17/bin/pip3.7 pip3.7 \
    && ln -s /opt/_internal/cpython-3.8.18/bin/pip3.8 pip3.8 \
    && ln -s /opt/_internal/cpython-3.9.18/bin/pip3.9 pip3.9 \
    && ln -s /opt/_internal/cpython-3.10.13/bin/pip3.10 pip3.10 \
    && ln -s python3.8 python3 \
    && ln -s pip3.8 pip3

RUN mkdir /root/.pip \
    && echo "[global]" > /root/.pip/pip.conf \
    && echo "index-url=https://mirrors.aliyun.com/pypi/simple" >> /root/.pip/pip.conf \
    && echo "trusted-host=mirrors.aliyun.com" >> /root/.pip/pip.conf \
    && echo "timeout=120" >> /root/.pip/pip.conf

RUN pip3.7 install pyyaml  \
    && pip3.7 install numpy==1.21.3 \
    && pip3.7 install protobuf==3.19.6 \
    && pip3.7 install decorator sympy scipy attrs cloudpickle psutil synr==0.5.0 tornado

RUN pip3.8 install pyyaml  \
    && pip3.8 install numpy==1.21.3 \
    && pip3.8 install protobuf==3.19.6 \
    && pip3.8 install decorator sympy scipy attrs cloudpickle psutil synr==0.5.0 tornado

RUN pip3.9 install pyyaml  \
    && pip3.9 install numpy==1.21.3 \
    && pip3.9 install protobuf==3.19.6 \
    && pip3.9 install decorator sympy scipy attrs cloudpickle psutil synr==0.5.0 tornado

RUN pip3.10 install pyyaml  \
    && pip3.10 install numpy==1.21.3 \
    && pip3.10 install protobuf==3.19.6 \
    && pip3.10 install decorator sympy scipy attrs cloudpickle psutil synr==0.5.0 tornado

RUN yum install -y java-1.8.0-openjdk wget blas blas-devel lapack lapack-devel ninja-build python2

WORKDIR /home

COPY build_protobuf.sh /home/
RUN bash /home/build_protobuf.sh
COPY install_obs.sh /home/
RUN bash /home/install_obs.sh
COPY install_cann.sh /home/
RUN bash /home/install_cann.sh
RUN echo "0 3 * * * /home/install_cann.sh" >> /etc/crontab