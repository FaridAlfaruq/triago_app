"""Laptop server workflow: init, check, status, run, demo, and share."""
import argparse
import base64
import json
import os
import secrets
import shutil
import subprocess
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from service.config import PROJECT_ROOT, load_config


def initialize():
    target = PROJECT_ROOT / ".env"
    if target.exists():
        print(".env sudah ada; file tidak diubah. Lihat config/laptop.env.example untuk opsi laptop.")
        return 1
    content = (PROJECT_ROOT / "config" / "laptop.env.example").read_text(encoding="utf-8")
    content = content.replace("replace-with-generated-dashboard-password", secrets.token_urlsafe(24))
    # Exclusive creation also prevents overwriting a file created concurrently.
    with target.open("x", encoding="utf-8") as output:
        output.write(content)
    print(".env laptop dibuat. Isi HOST, USERNAME, PASSWORD HiveMQ di file itu.")
    print("Password dashboard sudah dibuat secara acak dan hanya disimpan di .env.")
    return 0


def check_configuration():
    load_config()
    from backend.access_control import dashboard_credentials
    from service.mqtt_client import MqttConfig
    for key in ("TRIAGO_MQTT_HOST", "TRIAGO_MQTT_USERNAME", "TRIAGO_MQTT_PASSWORD"):
        value = os.environ.get(key, "")
        if not value or value.startswith(("replace-with-", "your-cluster")):
            raise ValueError(f"Isi {key} di .env; nilainya masih kosong/contoh")
    publisher = MqttConfig.from_env("publisher")
    backend = MqttConfig.from_env("backend")
    if publisher.client_id == backend.client_id:
        raise ValueError("Client ID publisher dan backend harus berbeda")
    credentials = dashboard_credentials()
    if not credentials:
        raise ValueError("Mode laptop publik memerlukan login dashboard; lihat config/laptop.env.example")
    if credentials[1].startswith("replace-with-") or len(credentials[1]) < 16:
        raise ValueError("Gunakan password dashboard acak minimal 16 karakter di .env")
    if os.environ.get("TRIAGO_HOST", "127.0.0.1") != "127.0.0.1":
        raise ValueError("Mode tunnel laptop memakai TRIAGO_HOST=127.0.0.1")
    if os.environ.get("TRIAGO_DEBUG", "0").lower() not in {"0", "false", "no"}:
        raise ValueError("Matikan TRIAGO_DEBUG untuk mode laptop publik")
    if os.environ.get("TRIAGO_MQTT_ENABLED", "0").lower() not in {"1", "true", "yes"}:
        raise ValueError("Aktifkan TRIAGO_MQTT_ENABLED=1 agar backend menerima MQTT")
    return credentials


def backend_url():
    port = int(os.environ.get("PORT", os.environ.get("TRIAGO_PORT", "5000")))
    return f"http://127.0.0.1:{port}"


def tunnel_binary():
    installed = shutil.which("cloudflared")
    if installed:
        return installed
    filename = "cloudflared.exe" if os.name == "nt" else "cloudflared"
    local = PROJECT_ROOT / "runtime" / "tools" / filename
    if local.is_file():
        return str(local)
    raise ValueError("cloudflared belum tersedia. Lihat docs/LAPTOP_SERVER.md untuk pemasangan")


def share():
    username, password = check_configuration()
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    endpoint = backend_url() + "/api/health"
    # Check the actual running service, not just the environment file.
    try:
        with urlopen(endpoint, timeout=5) as response:
            raise ValueError("Backend belum melindungi akses tanpa login; tunnel tidak dijalankan")
    except HTTPError as exc:
        if exc.code != 401:
            raise ValueError(f"Backend mengembalikan HTTP {exc.code}") from exc
    request = Request(endpoint, headers={"Authorization": "Basic " + token})
    with urlopen(request, timeout=5) as response:
        health = json.load(response)
    if not health.get("access", {}).get("login_required"):
        raise ValueError("Backend belum memakai login; tunnel tidak dijalankan")
    if not health.get("mqtt", {}).get("subscribed"):
        print("MQTT belum subscribe. Dashboard bisa dibuka, tetapi hasil MQTT belum dapat dipastikan masuk.")
    print("Membuka tunnel HTTPS. Bagikan URL trycloudflare yang muncul, serta login lewat jalur pribadi.")
    print("Biarkan terminal backend dan tunnel tetap berjalan. Ctrl+C menutup tunnel.")
    return subprocess.call([tunnel_binary(), "tunnel", "--url", backend_url()])


def status():
    username, password = check_configuration()
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    request = Request(backend_url() + "/api/health", headers={"Authorization": "Basic " + token})
    with urlopen(request, timeout=5) as response:
        health = json.load(response)
    print(json.dumps(health, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "check", "status", "run", "demo", "share"))
    args = parser.parse_args()
    try:
        if args.action == "init":
            return initialize()
        if args.action == "share":
            return share()
        if args.action == "status":
            return status()
        check_configuration()
        if args.action == "check":
            print("Konfigurasi laptop lengkap. Ini belum menguji login broker/database secara langsung.")
            return 0
        if args.action == "demo":
            from service.send_demo import main as send_demo
            return send_demo()
        from backend.app import app, socketio, SERVER_PORT
        print(f"Dashboard laptop: http://127.0.0.1:{SERVER_PORT} (login dari .env)")
        socketio.run(app, host="127.0.0.1", port=SERVER_PORT, debug=False,
                     use_reloader=False, allow_unsafe_werkzeug=True)
        return 0
    except (ValueError, OSError, URLError) as exc:
        print(f"Belum dapat menjalankan mode laptop: {exc}")
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
