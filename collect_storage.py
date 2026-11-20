"""Collect and archive the latest EIA natural-gas storage vintage."""
from juliany82.context_data import archive_storage_snapshot, fetch_eia_storage


def main() -> None:
    frame = fetch_eia_storage()
    path = archive_storage_snapshot(frame)
    print(f"Archived {len(frame)} EIA storage observations to {path}")


if __name__ == "__main__":
    main()
