FROM alpine:3.20
RUN apk add --no-cache bash curl
RUN curl -fsSL https://render.com/static/cli/install.sh | bash
ENV PATH="/root/.render:$PATH"
CMD ["bash", "-c", "render services resume facerecognition"]
