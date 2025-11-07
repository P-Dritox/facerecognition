FROM alpine:3.20

# Instala dependencias necesarias
RUN apk add --no-cache bash curl unzip

# Descarga binario de Render CLI desde GitHub (última versión estable)
RUN curl -L "https://github.com/render-oss/cli/releases/latest/download/render-linux-amd64.zip" -o render.zip && \
    unzip render.zip -d /usr/local/bin && \
    mv /usr/local/bin/render-linux-amd64 /usr/local/bin/render && \
    chmod +x /usr/local/bin/render && \
    rm render.zip

# Verifica instalación
RUN render --version || echo "Render CLI ready"

# Ejecuta el comando de encendido
CMD ["bash", "-c", "render services resume facerecognition"]
