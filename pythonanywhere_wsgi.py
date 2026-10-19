# ---------------------------------------------------------------------------
# PythonAnywhere WSGI configuration
# Copy everything below into:  Web tab -> "WSGI configuration file" link
#   (/var/www/<your-username>_pythonanywhere_com_wsgi.py)
# Replace YOUR_USERNAME with your PythonAnywhere username.
# ---------------------------------------------------------------------------
import sys

project_home = "/home/YOUR_USERNAME/churn_app"
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from app import app as application  # noqa: E402  (PythonAnywhere looks for "application")
