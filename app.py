# -*- coding: utf-8 -*-
"""Root entrypoint for Azure App Service deployment.

Forwarding Flask app & SocketIO from backend/app.py.
"""

import sys
import os

# Tambahkan direktori backend ke sys.path jika diperlukan
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app import app, socketio

if __name__ == "__main__":
    socketio.run(app)
