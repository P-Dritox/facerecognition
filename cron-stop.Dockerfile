FROM alpine:3.20

RUN apk add --no-cache bash curl unzip

RUN curl -L "https://github.com/render-oss/cli/releases/latest/download/render-linux-amd64.zip" -o render.zip && \
    unzip render.zip -d /usr/local/bin && \
    mv /usr/local/bin/render-linux-amd64 /usr/local/bin/render && \
    chmod +x /usr/local/bin/render && \
    rm render.zip

RUN render --version || echo "Render CLI ready"

CMD ["bash", "-c", "render services suspend facerecognition"]
