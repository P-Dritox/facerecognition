FROM alpine:3.20

# Instalar curl
RUN apk add --no-cache bash curl

# Comando principal
CMD ["bash", "-c", "curl -X POST https://api.render.com/v1/services/$SERVICE_ID/resume \
     -H \"Authorization: Bearer $RENDER_API_KEY\" \
     -H \"Accept: application/json\" && echo 'Service resumed'"]
