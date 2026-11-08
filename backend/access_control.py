"""Optional HTTP Basic login around Flask AND the Engine.IO middleware."""
import hmac
import os
from urllib.parse import urlsplit

from werkzeug.datastructures import Authorization
from werkzeug.wrappers import Response


def dashboard_credentials():
    username = os.environ.get("TRIAGO_DASHBOARD_USER", "")
    password = os.environ.get("TRIAGO_DASHBOARD_PASSWORD", "")
    required = os.environ.get("TRIAGO_REQUIRE_AUTH", "0").lower() in {"1", "true", "yes"}
    if required or username or password:
        if not username or not password:
            raise ValueError("Isi TRIAGO_DASHBOARD_USER dan TRIAGO_DASHBOARD_PASSWORD untuk login dashboard")
        return username, password
    return None


class DashboardLogin:
    def __init__(self, app, username, password):
        self.app = app
        self.username = username.encode("utf-8")
        self.password = password.encode("utf-8")

    def __call__(self, environ, start_response):
        auth = Authorization.from_header(environ.get("HTTP_AUTHORIZATION"))
        if not (auth and auth.type == "basic"
                and hmac.compare_digest((auth.username or "").encode("utf-8"), self.username)
                and hmac.compare_digest((auth.password or "").encode("utf-8"), self.password)):
            response = Response("Login dashboard diperlukan.", status=401,
                                headers={"WWW-Authenticate": 'Basic realm="TriaGO", charset="UTF-8"',
                                         "Cache-Control": "no-store"})
            return response(environ, start_response)
        # Browser form requests must not be able to reset/discharge beds cross-site.
        origin = environ.get("HTTP_ORIGIN")
        if origin and urlsplit(origin).netloc.lower() != environ.get("HTTP_HOST", "").lower():
            return Response("Origin tidak diizinkan.", status=403)(environ, start_response)

        def private_response(status, headers, exc_info=None):
            headers = [(name, value) for name, value in headers if name.lower() != "cache-control"]
            headers.extend([("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff")])
            return start_response(status, headers, exc_info)

        return self.app(environ, private_response)
