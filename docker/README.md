# Zen DNN Docker

## Build - dockerfiles

```
sudo docker build -f {DOCKER_FILE_NAME}.dockerfile -t {DOCKER_IMAGE_NAME} .
```

## Run - docker

```
sudo docker run -it --cap-add=SYS_RAWIO --network host --env DISPLAY=unix$DISPLAY --privileged --volume $XAUTH:/root/.Xauthority --volume /tmp/.X11-unix/:/tmp/.X11-unix {DOCKER_IMAGE_NAME}
```
