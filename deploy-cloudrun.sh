#!/bin/bash
# Usage: ./deploy-cloudrun.sh <gcp-project-id> <region> <service-name>
set -e
PROJECT_ID=$1
REGION=$2
SERVICE_NAME=$3
IMAGE=gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

gcloud builds submit --tag $IMAGE

gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --port 8080