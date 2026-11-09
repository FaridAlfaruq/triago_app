$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$toolsPath = Join-Path $projectRoot 'runtime/tools'
$destination = Join-Path $toolsPath 'cloudflared.exe'
if (Test-Path -LiteralPath $destination) {
    Write-Output 'cloudflared lokal sudah tersedia; file tidak diubah.'
    exit 0
}
New-Item -ItemType Directory -Force -Path $toolsPath | Out-Null
$release = Invoke-RestMethod -Uri 'https://api.github.com/repos/cloudflare/cloudflared/releases/latest'
$asset = $release.assets | Where-Object { $_.name -eq 'cloudflared-windows-amd64.exe' } | Select-Object -First 1
if (-not $asset -or $asset.digest -notmatch '^sha256:[a-fA-F0-9]{64}$') {
    throw 'Asset Windows atau checksum SHA256 resmi tidak tersedia; unduhan dibatalkan.'
}
$downloadPath = Join-Path $toolsPath 'cloudflared.download'
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $downloadPath
$actual = (Get-FileHash -LiteralPath $downloadPath -Algorithm SHA256).Hash
$expected = $asset.digest.Substring(7)
if ($actual -ne $expected) {
    throw 'Checksum tidak sesuai. File unduhan tidak dijalankan.'
}
Move-Item -LiteralPath $downloadPath -Destination $destination
Write-Output "cloudflared $($release.tag_name) tersimpan di runtime/tools; checksum SHA256 sesuai."
