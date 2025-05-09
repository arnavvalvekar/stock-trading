#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and print its output."""
    print(f"Running: {command}")
    process = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if process.returncode != 0:
        print(f"Error running command: {command}")
        print(process.stdout)
        sys.exit(1)
    return process.stdout

def setup_development_environment():
    """Set up the development environment."""
    # Get project root directory
    project_root = Path(__file__).parent.parent.absolute()
    
    # Create virtual environment
    print("Creating virtual environment...")
    run_command("python -m venv venv")
    
    # Activate virtual environment
    if sys.platform == "win32":
        activate_script = "venv\\Scripts\\activate"
    else:
        activate_script = "source venv/bin/activate"
    
    # Install dependencies
    print("Installing dependencies...")
    run_command(f"{activate_script} && pip install --upgrade pip")
    run_command(f"{activate_script} && pip install -r requirements.txt")
    
    # Install development dependencies
    print("Installing development dependencies...")
    run_command(f"{activate_script} && pip install -e .")
    
    # Create necessary directories
    print("Creating project directories...")
    directories = [
        "data/raw",
        "data/processed",
        "models/saved",
        "logs",
        "docs/build"
    ]
    for directory in directories:
        (project_root / directory).mkdir(parents=True, exist_ok=True)
    
    # Create .env file if it doesn't exist
    env_file = project_root / ".env"
    if not env_file.exists():
        print("Creating .env file...")
        with open(env_file, "w") as f:
            f.write("""# API Keys
ALPHA_VANTAGE_API_KEY=your_api_key_here

# Directories
MODEL_SAVE_DIR=models/saved
DATA_DIR=data
LOG_DIR=logs

# Logging
LOG_LEVEL=DEBUG
""")
    
    # Verify installation
    print("Verifying installation...")
    try:
        run_command(f"{activate_script} && python -c 'import stock_trading_ai'")
        print("Package installation verified successfully!")
    except Exception as e:
        print(f"Error verifying installation: {str(e)}")
        sys.exit(1)
    
    # Run tests
    print("Running tests...")
    run_command(f"{activate_script} && pytest tests/")
    
    print("\nDevelopment environment setup complete!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    print(f"   {activate_script}")
    print("2. Update the .env file with your API keys")
    print("3. Run the development server:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    setup_development_environment() 