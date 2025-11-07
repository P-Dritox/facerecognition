FROM alpine:3.20
RUN apk add --no-cache bash curl unzip
RUN curl -fsSL https://render.com/static/cli/install.sh -o /tmp/install.sh && \
    bash /tmp/install.sh && \
    mv /root/.render/render /usr/local/bin/render && \
    chmod +x /usr/local/bin/render
RUN render --version || echo "Render CLI installed."
CMD ["bash", "-c", "render services resume facerecognition"]
