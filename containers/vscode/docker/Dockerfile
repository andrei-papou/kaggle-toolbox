FROM docker.io/nvidia/cuda:11.6.2-base-ubuntu20.04

ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && \
	apt install -y software-properties-common && \
	add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update && apt install -y \
	curl \
	git \
	gcc \
	python3.10 \
	python3.10-dev \
	python3.10-venv \
	zip

# Override git editor
git config --global core.editor "code --wait"

# Install starship shell
RUN curl -sS https://starship.rs/install.sh > install_starship.sh && \
    chmod +x install_starship.sh && \
    ./install_starship.sh --yes \
    && echo "eval \"\$(starship init bash)\"" >> ~/.bashrc

RUN mkdir /root/.virtualenvs && mkdir /root/.requirements

COPY requirements /root/.requirements/kaggle-toolbox
RUN python3.10 -m venv /root/.virtualenvs/kaggle-toolbox
RUN PATH="/root/.virtualenvs/kaggle-toolbox/bin:$PATH" pip install -r /root/.requirements/kaggle-toolbox/local.txt
