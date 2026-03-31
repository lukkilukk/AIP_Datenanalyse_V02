param(
    [string]$InitialDirectory,
    [string]$FallbackDirectory
)

Set-StrictMode -Version Latest

Add-Type -AssemblyName System.Windows.Forms

$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Messdatei fuer AIP Datenanalyse auswaehlen'
$dialog.Filter = 'Textdateien (*.txt)|*.txt|Alle Dateien (*.*)|*.*'

$initialDir = $InitialDirectory
if ([string]::IsNullOrWhiteSpace($initialDir) -or -not (Test-Path -LiteralPath $initialDir)) {
    $initialDir = $FallbackDirectory
}

if (-not [string]::IsNullOrWhiteSpace($initialDir)) {
    $dialog.InitialDirectory = [System.IO.Path]::GetFullPath($initialDir)
}

$dialog.RestoreDirectory = $true

if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::WriteLine($dialog.FileName)
}
