#!/bin/bash
exec gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 --workers 4 --timeout 120