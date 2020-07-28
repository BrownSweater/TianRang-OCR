ARG PYTORCH="1.3"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ENV TZ=Asia/Shanghai
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN git submodule add git@git.tianrang-inc.com:algo-brain/gateway.git

COPY ./ /tianrang-ocr
RUN mkdir -p /tianrang-ocr/gateway/log
WORKDIR /tianrang-ocr

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 vim \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN pip --default-timeout=600 install --ignore-installed -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


EXPOSE 8333
RUN bash ./gateway/conf/start.sh