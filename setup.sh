#!/bin/bash

# Name of the virtual environment directory
VENV_DIR="venv"

# Create the virtual environment
python -m venv $VENV_DIR
echo "Virtual environment created at ./$VENV_DIR"

# Activate the virtual environment
source $VENV_DIR/bin/activate
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt

echo "Virtual environment activated."
source $VENV_DIR/bin/activate
#  streamlit run git-geneius.py