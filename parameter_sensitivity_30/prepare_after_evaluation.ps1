$ErrorActionPreference = 'Stop'
$taskRoot = 'H:\RadarST_FigureRevision2_20261111'
$scientificPython = 'C:\Environment_Langguage_Software\Anaconda\envs\st_benchmark\python.exe'
$runtimeRoot = 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt\dependencies'
$documentPython = "$runtimeRoot\python\python.exe"
$artifactNode = "$runtimeRoot\node\bin\node.exe"
function Invoke-Checked {
    param([string]$taskExecutable, [string[]]$taskArguments)
    & $taskExecutable @taskArguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $taskExecutable $taskArguments" }
}
while (-not (Test-Path -LiteralPath "$taskRoot\evaluation\EVALUATION_COMPLETE.json")) {
    if (Test-Path -LiteralPath "$taskRoot\qa\PREDICTION_FAILURES.json") { throw 'Prediction failure; no final preparation.' }
    if ((Test-Path -LiteralPath "$taskRoot\logs\EVALUATE_ALL5040.log") -and (Select-String -LiteralPath "$taskRoot\logs\EVALUATE_ALL5040.log" -Pattern '^Traceback' -Quiet)) { throw 'Evaluation error; no final preparation.' }
    Start-Sleep -Seconds 5
}
Invoke-Checked $scientificPython @('-c', "import sys;sys.path.insert(0,'H:/RadarST_FigureRevision2_20261111');import revise_figures as f;f.figure5();f.export('Figure5');f.export('Figure5_all322_maps')")
Invoke-Checked $scientificPython @("$taskRoot\verify_final_artifacts.py")
Invoke-Checked $scientificPython @("$taskRoot\audit_figures.py", 'Figure5')
if (-not (Test-Path -LiteralPath "$taskRoot\qa\XLSX_ARTIFACT_MARKER_DONE.txt")) {
    Invoke-Checked $artifactNode @('C:\Users\Administrator\.codex\plugins\cache\openai-primary-runtime\spreadsheets\26.1104.22356\skills\spreadsheets\container_tools\mark_artifact_operation_started.mjs', '--operation-kind', 'edit', '--expected-output-count', '1', '--output-format', 'xlsx')
    Set-Content -LiteralPath "$taskRoot\qa\XLSX_ARTIFACT_MARKER_DONE.txt" -Value 'SUCCESS' -Encoding utf8
}
Invoke-Checked $artifactNode @("$taskRoot\build_s31.mjs")
Invoke-Checked $documentPython @("$taskRoot\merge_s31.py")
if (-not (Test-Path -LiteralPath "$taskRoot\qa\DOCX_ARTIFACT_MARKER_DONE.txt")) {
    Invoke-Checked $artifactNode @('C:\Users\Administrator\.codex\plugins\cache\openai-primary-runtime\documents\26.1104.22356\skills\documents\container_tools\mark_artifact_operation_started.mjs', '--operation-kind', 'edit', '--expected-output-count', '3', '--output-format', 'docx')
    Set-Content -LiteralPath "$taskRoot\qa\DOCX_ARTIFACT_MARKER_DONE.txt" -Value 'SUCCESS' -Encoding utf8
}
Invoke-Checked $documentPython @("$taskRoot\stage_documents.py")
Invoke-Checked $documentPython @("$taskRoot\render_and_check_documents.py")
Set-Content -LiteralPath "$taskRoot\qa\PREPARED_FOR_MANUAL_REVIEW.txt" -Value 'All results and artifacts prepared; manual visual review required before publication.' -Encoding utf8
