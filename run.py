import os
import sys
import uvicorn
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config.config import get_api_config

if __name__ == "__main__":
    # Get API configuration
    api_config = get_api_config()
    
    # Run the API server
    uvicorn.run(
        "app.api.api:app",
        host=api_config["host"],
        port=api_config["port"],
        reload=api_config["reload"]
    ) 