FROM registry.giaohangtietkiem.vn/nlp/nlp-as-service/base-image-nvidia:v1.0.0

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install torch_geometric
RUN pip3 install scipy==1.10.1
RUN pip3 install rdkit
RUN pip3 install wandb

COPY data/ /root/MSGC-DTA
WORKDIR /code/MSGC-DTA
COPY src/ /code/MSGC-DTA/
