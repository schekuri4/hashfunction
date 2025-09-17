@echo off
echo === Rebuilding GPU Optimizer with All Datasets ===
echo.
echo Setting up Visual Studio environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
echo.
echo Compiling GPU optimizer...
nvcc -O3 -o hash_optimizer_gpu_infinite.exe hash_optimizer_gpu_infinite.cu -lcurand
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ GPU optimizer rebuilt successfully!
    echo ✓ Now includes all 13 input datasets
    echo.
    echo To test: .\hash_optimizer_gpu_infinite.exe
) else (
    echo.
    echo ✗ GPU optimizer build failed!
    echo Make sure CUDA Toolkit and Visual Studio are properly installed.
)
pause