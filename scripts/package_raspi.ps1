$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$packageDir = Join-Path $projectRoot 'runtime/packages'
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null
$manifestPath = Join-Path $packageDir 'raspi-manifest.txt'
$archivePath = Join-Path $packageDir 'triago-raspi-mqtt.tar.gz'
$files = @(
    'service/__init__.py',
    'service/api_client.py',
    'service/config.py',
    'service/mqtt_client.py',
    'service/mqtt_queue.py',
    'service/send_demo.py',
    'service/triage_payload.py',
    'config/raspi.env.example',
    'requirements-mqtt.txt',
    'docs/RASPI_MQTT.md'
)
# Explicit allowlist: never include .env, runtime queues, or measurement data.
foreach ($relativePath in $files) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $relativePath))) {
        throw "File paket tidak ditemukan: $relativePath"
    }
}
[IO.File]::WriteAllLines($manifestPath, $files, [Text.UTF8Encoding]::new($false))
& tar -C $projectRoot -czf $archivePath -T $manifestPath
if ($LASTEXITCODE -ne 0) { throw 'Gagal membuat arsip Raspi' }
Write-Output $archivePath
