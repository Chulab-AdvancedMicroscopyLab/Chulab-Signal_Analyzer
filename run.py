import os
import subprocess
import sys
from pathlib import Path

def main():
    """
    A cross-platform script to set up and run the zarr_analyzer Docker container.

    This script performs the following actions:
    1. Prompts the user for the path to their data directory.
    2. Validates that the provided path exists.
    3. Generates a `docker-compose.yml` file dynamically, configured to mount
       the user's data and the project's `utils` directory into the container.
    4. Starts the Docker services using `docker compose up --build`.
    5. On exit (including Ctrl+C), it gracefully stops the services with
       `docker compose down` and cleans up the generated `docker-compose.yml` file.
    """
    # --- 1. Prompt user for the data directory ---
    try:
        user_input = input("Enter the path to your data directory (e.g., D:/path/to/data): ")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting.")
        sys.exit(0)

    # --- 2. Normalize and validate the path ---
    host_data_path = Path(user_input).resolve()

    if not host_data_path.exists():
        print(f"Error: The specified path does not exist: {host_data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Path validated: {host_data_path}")

    # --- 3. Define Docker configuration ---
    project_root = Path(__file__).parent.resolve()
    compose_file_path = project_root / "docker-compose.yml"

    # Docker variables
    container_name = "zarr_analyzer"
    container_workspace_path = "/workspace"
    
    # Convert paths to a string format with forward slashes for Docker compatibility
    host_data_path_str = str(host_data_path).replace('\\', '/')
    container_data_path = f"{container_workspace_path}/datas"

    host_utils_path_str = str(project_root / "utils").replace('\\', '/')
    container_utils_path = f"{container_workspace_path}/utils"

    # --- 4. Generate docker-compose.yml ---
    # Using an f-string with triple quotes for a clean YAML template
    docker_compose_content = f"""
services:
  {container_name}:
    container_name: {container_name}
    build:
      context: ./
      dockerfile: Dockerfile
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTHONUNBUFFERED=1
      - PYTHONDONTWRITEBYTECODE=1
    volumes:
      - "{host_data_path_str}:{container_data_path}"
      - "{host_utils_path_str}:{container_utils_path}"
    working_dir: {container_workspace_path}
    command: bash
    stdin_open: true
    tty: true
"""

    print("Generating docker-compose.yml...")
    with open(compose_file_path, "w") as f:
        f.write(docker_compose_content.strip())

    # --- 5. Run Docker Compose and handle cleanup ---
    try:
        print("Starting container via 'docker compose up --build'...")
        print("Press Ctrl+C to stop the container and clean up.")
        # Use shell=True on Windows to ensure `docker` is found if not in system PATH
        subprocess.run(["docker", "compose", "up", "--build"], check=True, shell=sys.platform == "win32")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running Docker Compose: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nUser interrupted. Stopping container...")
    finally:
        print("Stopping and cleaning up Docker container...")
        subprocess.run(["docker", "compose", "down"], check=False, shell=sys.platform == "win32")
        if compose_file_path.exists():
            os.remove(compose_file_path)
            print("Removed generated docker-compose.yml")

if __name__ == "__main__":
    main()
