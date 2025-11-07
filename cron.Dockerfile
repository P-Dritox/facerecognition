FROM alpine:3.20
RUN apk add --no-cache bash curl

CMD ["bash", "-c", "\
  echo 'Resuming service...'; \
  curl -X POST https://api.render.com/v1/services/$SERVICE_ID/resume \
  -H 'Authorization: Bearer '$RENDER_API_KEY \
  -H 'Accept: application/json' && \
  echo '\n✅ Service resumed successfully.'"]
