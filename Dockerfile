FROM ubuntu:latest
LABEL authors="olehpron"

ENTRYPOINT ["top", "-b"]