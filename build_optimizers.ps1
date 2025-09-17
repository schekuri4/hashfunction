# Hash Function Optimizer Build Script

Write-Host "=== Building Hash Function Optimizers ===" -ForegroundColor Green

# Build CPU version
Write-Host "`nBuilding CPU optimizer..." -ForegroundColor Yellow
try {
    g++ -std=c++11 -O3 -pthread -o hash_optimizer_cpu hash_optimizer.cpp
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ CPU optimizer built successfully!" -ForegroundColor Green
    } else {
        Write-Host "✗ CPU optimizer build failed!" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Error building CPU optimizer: $_" -ForegroundColor Red
}

# Check if NVCC is available for GPU version
Write-Host "`nChecking for CUDA compiler..." -ForegroundColor Yellow
$nvccAvailable = $false
try {
    $nvccVersion = nvcc --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ NVCC found!" -ForegroundColor Green
        $nvccAvailable = $true
    }
} catch {
    Write-Host "✗ NVCC not found in PATH" -ForegroundColor Red
}

# Build GPU version if NVCC is available
if ($nvccAvailable) {
    Write-Host "`nBuilding GPU optimizer..." -ForegroundColor Yellow
    try {
        nvcc -O3 -o hash_optimizer_gpu hash_optimizer_gpu.cu -lcurand
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ GPU optimizer built successfully!" -ForegroundColor Green
        } else {
            Write-Host "✗ GPU optimizer build failed!" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ Error building GPU optimizer: $_" -ForegroundColor Red
    }
} else {
    Write-Host "`nSkipping GPU optimizer build (CUDA not available)" -ForegroundColor Yellow
    Write-Host "To build GPU version:" -ForegroundColor Cyan
    Write-Host "1. Install CUDA Toolkit from NVIDIA" -ForegroundColor Cyan
    Write-Host "2. Add nvcc to your PATH" -ForegroundColor Cyan
    Write-Host "3. Run: nvcc -O3 -o hash_optimizer_gpu hash_optimizer_gpu.cu -lcurand" -ForegroundColor Cyan
}

# Create a simple launcher script
Write-Host "`nCreating launcher script..." -ForegroundColor Yellow

Write-Host "`n=== Build Complete ===" -ForegroundColor Green
Write-Host "To run optimizers, use: .\run_optimizer.ps1" -ForegroundColor Cyan
Write-Host "Or run directly:" -ForegroundColor Cyan
if (Test-Path "hash_optimizer_cpu.exe") {
    Write-Host "  .\hash_optimizer_cpu.exe" -ForegroundColor White
} elseif (Test-Path "hash_optimizer_cpu") {
    Write-Host "  .\hash_optimizer_cpu" -ForegroundColor White
}
if (Test-Path "hash_optimizer_gpu.exe") {
    Write-Host "  .\hash_optimizer_gpu.exe" -ForegroundColor White
} elseif (Test-Path "hash_optimizer_gpu") {
    Write-Host "  .\hash_optimizer_gpu" -ForegroundColor White
}