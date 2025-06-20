# Requer execução como Administrador

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

$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
$cudaBinPath = Join-Path $cudaPath "bin"

# Adicionar CUDA ao PATH do sistema
$currentPath = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::Machine)
if (-not $currentPath.Contains($cudaBinPath)) {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$cudaBinPath", [EnvironmentVariableTarget]::Machine)
    Write-Host "CUDA bin path adicionado ao PATH do sistema" -ForegroundColor Green
}

# Verificar e copiar DLLs
foreach ($dll in $dllsToCheck) {
    $sourcePath = Join-Path $cudaBinPath $dll
    if (Test-Path $sourcePath) {
        Write-Host "$dll encontrado em $cudaBinPath" -ForegroundColor Green
        
        # Copiar para System32 (pode ser necessário para alguns aplicativos)
        $system32Path = "C:\Windows\System32\$dll"
        if (-not (Test-Path $system32Path)) {
            Copy-Item $sourcePath $system32Path -Force
            Write-Host "$dll copiado para System32" -ForegroundColor Yellow
        }
    } else {
        Write-Host "$dll NÃO encontrado em $cudaBinPath" -ForegroundColor Red
    }
}

# Criar/Atualizar variáveis de ambiente CUDA
[Environment]::SetEnvironmentVariable("CUDA_PATH", $cudaPath, [EnvironmentVariableTarget]::Machine)
[Environment]::SetEnvironmentVariable("CUDA_PATH_V11_2", $cudaPath, [EnvironmentVariableTarget]::Machine)