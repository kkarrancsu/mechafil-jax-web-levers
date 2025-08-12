#!/bin/bash

gcloud builds submit --tag gcr.io/cel-streamlit/mechafil-jax-web-levers --no-cache

gcloud run deploy mechafil-jax-web-levers \
  --image gcr.io/cel-streamlit/mechafil-jax-web-levers \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances=0 \
  --port=8501
