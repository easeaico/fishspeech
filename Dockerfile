FROM nvidia/cuda:12.2.2-runtime-rockylinux9

RUN dnf install -y --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm
RUN dnf install -y --nogpgcheck https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-$(rpm -E %rhel).noarch.rpm
RUN dnf -y update && dnf install openssl-devel bzip2-devel libffi-devel zlib-devel wget git xz xz-devel cmake gcc gcc-c++ -y && dnf clean all 
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tar.xz && tar -xf Python-3.10.14.tar.xz
RUN cd Python-3.10.14 && ./configure --enable-optimizations && make -j 2 && nproc && make altinstall
RUN pip3.10 install --no-cache-dir --upgrade pip

WORKDIR /work
COPY . .

ENV CFLAGS="-fPIC"
RUN pip3.10 install torch torchvision torchaudio
RUN pip3.10 install --no-cache-dir -e .

EXPOSE 8051

VOLUME /work/checkpoints
VOLUME /work/references

CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8051"]
