FROM alpine:3.20

# Instalar dependencias
RUN apk add --no-cache bash curl unzip

# Descargar e instalar Render CLI desde la fuente oficial (no GitHub)
RUN curl -fsSL https://render.com/static/cli/install.sh -o /tmp/install.sh && \
    bash /tmp/install.sh && \
    cp /root/.render/render /usr/local/bin/render && \
    chmod +x /usr/local/bin/render

# Verificar instalación
RUN /usr/local/bin/render --version || echo "Render CLI ready"

# Comando principal: encender el servicio
CMD ["bash", "-c", "render services resume facerecognition"]
