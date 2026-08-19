#!/bin/bash
echo "Starting Dataset Pipeline..."
python generate_dataset.py
python validate_dataset.py
echo "Pipeline Completed!"
