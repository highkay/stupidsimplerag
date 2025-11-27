from dotenv import find_dotenv, load_dotenv

# Load configuration from .env when the package is imported, so uvicorn
# does not need --env-file during local development.
load_dotenv(find_dotenv(), override=False)
