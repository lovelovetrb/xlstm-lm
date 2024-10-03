FROM continuumio/anaconda3

# 作業ディレクトリを設定
WORKDIR /app

COPY environment.yaml /app/environment.yaml

# 環境を作成し、初期化する
RUN conda env create -n xlstm -f /app/environment.yaml && \
    conda init bash && \
    echo "conda activate xlstm" >> ~/.bashrc

RUN conda install --channel conda-forge nvtop

# 環境変数を設定してxlstm環境のパスを指定
ENV PATH=/opt/conda/envs/xlstm/bin:$PATH

RUN pip install xlstm \
    transformers \
    datasets \ 
    wandb


ENV PYTHONPATH=/app:/app/src:$PYTHONPATH

COPY .netrc /root/.netrc

# デフォルトのシェルをbashに設定し、環境をアクティブ化
SHELL ["/bin/bash", "--login", "-c"]

