from sqlalchemy import create_engine

from logistics_ml.config.database import database

engine = create_engine(database.url)
