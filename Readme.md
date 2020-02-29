# Running Instructions

1) All images must be of .jpg format and of having minimum size of 224*224
2) clone this repository and cd to this directory. alternatively you can directly build image using git repo.
3) Folder named "test_files" must be created and all .jpg files must be copied in folder for processing.
4) Run below given command to build docker image
```
docker build -t philips-challenge-scoring .
```
5) run docker image by typing below code snippet
```
docker run philips-challenge-scoring
```