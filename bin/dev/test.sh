PYTHONPATH="$(pwd):$PYTHONPATH" pytest && mypy && flake8
