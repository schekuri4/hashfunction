# Hash Function Optimizer Launcher
Write-Host "=== Hash Function Seed Optimizer ===" -ForegroundColor Green

Write-Host "`nThis program uses CPU multi-threading to find optimal hash function seeds" -ForegroundColor Yellow
Write-Host "It will test random combinations of h and k seed values to minimize standard deviation" -ForegroundColor Yellow

Write-Host "`nRecommended test amounts:" -ForegroundColor Cyan
Write-Host "- Quick test: 10,000 tests (~30 seconds)" -ForegroundColor White
Write-Host "- Medium test: 100,000 tests (~5 minutes)" -ForegroundColor White  
Write-Host "- Thorough test: 1,000,000+ tests (~30+ minutes)" -ForegroundColor White

Write-Host "`nCurrent best known result:" -ForegroundColor Yellow
if (Test-Path "optimization_results.txt") {
    $results = Get-Content "optimization_results.txt"
    foreach ($line in $results) {
        Write-Host "  $line" -ForegroundColor White
    }
} else {
    Write-Host "  No previous results found" -ForegroundColor Gray
}

Write-Host "`nStarting optimizer..." -ForegroundColor Green
.\hash_optimizer_cpu.exe