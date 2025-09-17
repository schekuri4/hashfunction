# Test all input files and calculate average standard deviation
Write-Host "=== Testing All Input Files and Calculating Average ===" -ForegroundColor Green

# Array to store all standard deviations
$standardDeviations = @()

# List of input files
$inputFiles = @("sample_input.txt", "atoz.txt", "common500.txt", "bertuncased.txt", "mit_a.txt")

# Test each file
foreach ($file in $inputFiles) {
    Write-Host "`nTesting $file..." -ForegroundColor Cyan
    
    # Get the standard deviation (last line of output)
    $output = Get-Content "inputs\$file" | .\encoder.exe
    $stdDev = $output[-1]
    
    Write-Host "$file : $stdDev" -ForegroundColor Yellow
    
    # Add to array (convert to double for calculation)
    $standardDeviations += [double]$stdDev
}

# Calculate average
$average = ($standardDeviations | Measure-Object -Average).Average

Write-Host "`n=== RESULTS ===" -ForegroundColor Green
Write-Host "Individual Standard Deviations:" -ForegroundColor White
for ($i = 0; $i -lt $inputFiles.Length; $i++) {
    Write-Host "  $($inputFiles[$i]): $($standardDeviations[$i])" -ForegroundColor White
}

Write-Host "`nAverage Standard Deviation: $($average.ToString('F5'))" -ForegroundColor Magenta