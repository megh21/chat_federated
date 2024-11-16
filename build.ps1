# Define variables
$imageName = "mbnextsolutions/chatresnextsol"
$tag = "latest"

# Navigate to the project directory
Set-Location "D:\project\chat_federated"

# Build the Docker image
docker build -t ${imageName}:${tag} .
