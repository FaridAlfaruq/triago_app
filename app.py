# -*- coding: utf-8 -*-
"""Root entrypoint for the TriaGO backend.

Forwarding Flask app & SocketIO from backend/app.py.
"""

import sys
import os

# Tambahkan direktori backend ke sys.path jika diperlukan
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app import app, socketio, SERVER_HOST, SERVER_PORT, DEBUG_MODE

if __name__ == "__main__":
    socketio.run(app, host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG_MODE,
                 use_reloader=False, allow_unsafe_werkzeug=True)
