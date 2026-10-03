import os

# Tests never export traces, even when .env holds real Langfuse keys (load_dotenv won't override).
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
