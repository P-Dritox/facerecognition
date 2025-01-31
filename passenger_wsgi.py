import sys
import os

APP_DIR = '/home/kaizenapps/myapp'
VENV_DIR = '/home/kaizenapps/virtualenv/myapp/3.11/lib/python3.11/site-packages'

sys.path.insert(0, APP_DIR)
sys.path.insert(0, VENV_DIR)

from app import app as application
