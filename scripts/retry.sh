#!/bin/bash

MAX_RETRIES=3
RETRY_DELAY=2
COUNTER=0

while [ $COUNTER -lt $MAX_RETRIES ]; do
  eval "$@" && break
  COUNTER=$((COUNTER+1))
  if [ $COUNTER -lt $MAX_RETRIES ]; then
    echo "Command failed. Retrying in $RETRY_DELAY seconds..."
    sleep $RETRY_DELAY
  else
    echo "Command failed after $COUNTER attempts."
    exit 1
  fi
done

