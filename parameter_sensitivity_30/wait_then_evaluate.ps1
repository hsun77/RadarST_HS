$ErrorActionPreference = 'Stop'
$taskRoot = 'H:\RadarST_FigureRevision2_20261111'
while (-not (Test-Path -LiteralPath "$taskRoot\ALL5040_PREDICTIONS_LOCKED.json")) {
    if (Test-Path -LiteralPath "$taskRoot\qa\PREDICTION_FAILURES.json") { throw 'Prediction failures recorded; evaluation not started.' }
    $running = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*FigureRevision2_20261111*sensitivity.py*predict*' }
    if (-not $running) { throw 'Prediction process stopped before the global lock; evaluation not started.' }
    Start-Sleep -Seconds 5
}
& 'C:\Environment_Langguage_Software\Anaconda\envs\st_benchmark\python.exe' -u "$taskRoot\evaluate_sensitivity.py" --workers 8 *> "$taskRoot\logs\EVALUATE_ALL5040.log"
exit $LASTEXITCODE
