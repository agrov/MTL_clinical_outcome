FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

ENV PYTHONIOENCODING=utf-8

RUN git clone https://github.com/deepset-ai/FARM.git
RUN cd FARM; pip install -r requirements.txt 
RUN cd FARM; git checkout 59358bc18887e17835cc9953b77ee22887d47a6b; pip install .

RUN pip install ray[tune]
RUN pip install hyperopt
RUN pip install fire
RUN pip install dotmap==1.3.0

