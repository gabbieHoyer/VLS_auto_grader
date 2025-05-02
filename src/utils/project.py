# src/utils/experiment_setup.py
import os
import pyrootutils

def get_project_root():
    """Find the project root directory based on .git indicator."""
    return pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git"],
        pythonpath=True,
        dotenv=True,
    )
