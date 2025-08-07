#!/bin/bash
# Script para iniciar Jupyter Lab no ambiente virtual

cd /home/robca/obd2
source venv/bin/activate

echo "🐍 Ativando ambiente virtual..."
echo "Python: $(which python)"
echo "Jupyter: $(which jupyter)"

echo ""
echo "🚀 Iniciando Jupyter Lab..."
echo "📍 URL será exibida abaixo - copie o token para conectar no VS Code"
echo ""

jupyter lab --port=8888 --no-browser --ip=127.0.0.1 --allow-root
