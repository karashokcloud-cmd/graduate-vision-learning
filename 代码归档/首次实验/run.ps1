# 先按README在VS Code里创建 train_transfer.py，再运行本入口。
$taskScript = Join-Path $PSScriptRoot 'train_transfer.py'
if (-not (Test-Path -LiteralPath $taskScript)) {
    Write-Host '请先在VS Code新建 train_transfer.py，复制课程代码中的参考代码并保存。'
    exit 1
}
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') $taskScript
