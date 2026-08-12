import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DatabaseConfig:

    url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://logistics:logistics@localhost:5432/logistics",
    )


database = DatabaseConfig()
