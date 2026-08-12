import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KafkaConfig:

    bootstrap_servers: str = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "localhost:9092",
    )

    trips_topic: str = os.getenv(
        "KAFKA_TRIPS_TOPIC",
        "taxi-trips",
    )

    features_topic: str = os.getenv(
        "KAFKA_FEATURES_TOPIC",
        "taxi-features",
    )


kafka = KafkaConfig()
