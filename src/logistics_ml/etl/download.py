import urllib.request

from logistics_ml.config import data as data_config


def download_taxi_lookup():
    data_config.raw_data_dir.mkdir(parents=True, exist_ok=True)

    lookup_target = data_config.taxi_lookup
    if lookup_target.exists():
        print(f"✓ {lookup_target} already exists.")
        return

    lookup_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    print(f"Downloading {lookup_url}")
    urllib.request.urlretrieve(lookup_url, lookup_target)
    print(f"✓ Saved to {lookup_target}")


def download_data():
    data_config.raw_data_dir.mkdir(parents=True, exist_ok=True)

    for url in data_config.taxi_urls:
        filename = url.split("/")[-1]
        target = data_config.raw_data_dir / filename

        if target.exists():
            print(f"✓ {target} already exists.")
            continue

        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, target)
        print(f"✓ Saved to {target}")

    download_taxi_lookup()


def main():
    download_data()
    download_taxi_lookup()


if __name__ == "__main__":
    main()
