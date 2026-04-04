from __future__ import annotations

from panda_matching.ingest.blackandwhitebear import load_profiles


def main() -> None:
    loaded = load_profiles()
    print(f"Loaded {loaded} panda profiles from blackandwhitebear.com")


if __name__ == "__main__":
    main()
