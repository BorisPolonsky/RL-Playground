FROM tensorflow/tensorflow:latest-gpu-py3
RUN apt update && apt install -y python-pyglet python3-opengl zlib1g-dev libjpeg-dev patchelf cmake swig libboost-all-dev libsdl2-dev libosmesa6-dev xvfb ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip3 install gym[atari]
WORKDIR /apps
CMD /bin/bash
