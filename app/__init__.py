from dotenv import find_dotenv, load_dotenv

from app.logging_config import configure_logging

# Load configuration from .env when the package is imported, so uvicorn
# does not need --env-file during local development.
load_dotenv(find_dotenv(), override=False)

# Apply LOG_LEVEL / APP_LOG_LEVEL overrides as soon as env is available.
configure_logging()
