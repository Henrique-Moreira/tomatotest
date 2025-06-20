$dllsToCheck = @(
    "cudart64_110.dll",
    "cublas64_11.dll",
    "cublasLt64_11.dll",
    "cufft64_10.dll",
    "curand64_10.dll",
    "cusolver64_11.dll",
    "cusparse64_11.dll",
    "cudnn64_8.dll"
)

$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"

foreach ($dll in $dllsToCheck) {
    $dllPath = Join-Path $cudaPath $dll
    if (Test-Path $dllPath) {
        Write-Host "$dll encontrado em $cudaPath" -ForegroundColor Green
    } else {
        Write-Host "$dll NÃO encontrado em $cudaPath" -ForegroundColor Red
    }
}