# ============================================================
# FTMO Bot — Instalacion automatica en VPS
# Ejecutar en PowerShell como Administrador dentro del VPS
# ============================================================

$ErrorActionPreference = "Stop"
$botDir = "C:\ftmo_bot"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  FTMO Bot - Instalacion VPS" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# 1. Verificar/instalar Python
Write-Host "`n[1/4] Verificando Python..." -ForegroundColor Yellow
$pythonOk = $false
try {
    $ver = python --version 2>&1
    if ($ver -match "Python 3") {
        Write-Host "  Python ya instalado: $ver" -ForegroundColor Green
        $pythonOk = $true
    }
} catch {}

if (-not $pythonOk) {
    Write-Host "  Descargando Python 3.11..." -ForegroundColor Yellow
    $pyInstaller = "$env:TEMP\python_installer.exe"
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile $pyInstaller
    Write-Host "  Instalando Python (silencioso)..." -ForegroundColor Yellow
    Start-Process $pyInstaller -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" -Wait
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
    Write-Host "  Python instalado." -ForegroundColor Green
}

# 2. Clonar repo de GitHub
Write-Host "`n[2/4] Descargando archivos del bot..." -ForegroundColor Yellow
if (Test-Path $botDir) {
    Write-Host "  Actualizando carpeta existente..." -ForegroundColor Yellow
    Set-Location $botDir
    git pull
} else {
    git clone https://github.com/Pablo-app-developer/BOT_ML_FTMO_10000.git $botDir
    Set-Location $botDir
}
Write-Host "  Archivos descargados en $botDir" -ForegroundColor Green

# 3. Instalar dependencias Python
Write-Host "`n[3/4] Instalando dependencias Python..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
python -m pip install MetaTrader5 pandas numpy requests --quiet
Write-Host "  Dependencias instaladas." -ForegroundColor Green

# 4. Crear acceso directo en el escritorio
Write-Host "`n[4/4] Creando acceso directo en el escritorio..." -ForegroundColor Yellow
$desktop = [Environment]::GetFolderPath("CommonDesktopDirectory")
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut("$desktop\FTMO BOT.lnk")
$shortcut.TargetPath = "$botDir\ARRANCAR_BOT.bat"
$shortcut.WorkingDirectory = $botDir
$shortcut.Save()
Write-Host "  Acceso directo creado en el escritorio." -ForegroundColor Green

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  INSTALACION COMPLETA" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "PASOS FINALES (hazlos tu):" -ForegroundColor White
Write-Host "  1. Instala MetaTrader 5 desde: https://www.metatrader5.com/es/download" -ForegroundColor White
Write-Host "  2. Abre MT5, conéctate a tu cuenta FTMO" -ForegroundColor White
Write-Host "  3. Activa AutoTrading (botón verde arriba en MT5)" -ForegroundColor White
Write-Host "  4. Doble clic en 'FTMO BOT' en el escritorio" -ForegroundColor White
Write-Host ""
Write-Host "El bot arranca. Puedes desconectar el RDP — sigue corriendo." -ForegroundColor Green
Write-Host ""
Read-Host "Presiona Enter para cerrar"
