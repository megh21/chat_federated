# Define variables
$imageName = "mbnextsolutions/chatresnextsol"
$tag = "latest"

# Navigate to the project directory
Set-Location "D:\project\chat_federated"

# Build the Docker image
docker build -t ${imageName}:${tag} .

# # Log in to Docker Hub using environment variables
# docker login --username mbnextsolutions --password madman213
# # Push the Docker image to Docker Hub
# docker push ${imageName}:${tag}

# # Log out from Docker Hub
# docker logout