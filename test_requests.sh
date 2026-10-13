#!/usr/bin/env bash
# Multipart
curl -X POST http://127.0.0.1:8000/api/v1/classify -F "image=@./sample.jpg"

# JSON/Base64 (requiere jq y sample.jpg)
BASE64=$(base64 -w 0 ./sample.jpg)
curl -X POST http://127.0.0.1:8000/api/v1/classify \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"data:image/jpeg;base64,$BASE64\",\"filename\":\"sample.jpg\"}"
