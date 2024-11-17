# Define variables
$imageName = "mbnextsolutions/chatresnextsol"
$tag = "latest"

# Navigate to the project directory which is current directory
Set-Location $PSScriptRoot

# Build the Docker image
docker build -t ${imageName}:${tag} .
