FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y wget ca-certificates \
    git curl vim python3-dev python3-pip \
    libfreetype6-dev
RUN pip3 install --upgrade pip
RUN pip3 install h5py jupyter matplotlib numpy pandas pyyaml seaborn sklearn statsmodels
RUN pip3 install keras --no-deps
RUN pip3 install keras_applications --no-deps
RUN pip3 install keras_preprocessing --no-deps
RUN pip3 install tabulate
RUN pip3 install tensorflow-gpu
RUN pip3 install tqdm

RUN ["mkdir", "notebooks"]
COPY jupyter_notebook_config.py /root/.jupyter/
COPY run_jupyter.sh
VOLUME /notebooksCMD [“/run_jupyter.sh”]
