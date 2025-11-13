#!/bin/bash

echo "Corrigiendo imports..."

# Corregir imports en todo el directorio app/
find app/ -type f -name "*.py" -exec sed -i '' \
  -e 's/^from database import/from app.database import/g' \
  -e 's/^from models import/from app.models import/g' \
  -e 's/^from jobs import/from app.jobs import/g' \
  -e 's/^from utils\./from app.utils./g' \
  -e 's/^from routes\./from app.routes./g' \
  -e 's/^from migrations import/from app.migrations import/g' \
  {} \;

echo "Imports corregidos!"
echo "Archivos modificados:"
git diff --name-only
