#!/bin/bash
set -e

echo "Running black..."
black .

echo "Running isort..."
isort .

echo "Running flake8..."
flake8 src tests

echo "Running mypy..."
mypy src tests

echo "All linting checks passed!"