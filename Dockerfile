FROM tensorflow/tensorflow:latest-gpu-py3

ADD start.py /

RUN pip install gpt_2_simple

CMD [ "python", "./start.py" ]