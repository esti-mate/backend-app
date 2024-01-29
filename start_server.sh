# Activate the virtual environment
source venv/bin/activate
gunicorn -w 4 -b :3003 app:app