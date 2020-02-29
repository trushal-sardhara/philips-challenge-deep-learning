FROM python:3

ADD score_file.py /
ADD final_model.pth /
ADD test_files /test_files

RUN pip install torch torchvision

CMD [ "python", "./score_file.py" ]