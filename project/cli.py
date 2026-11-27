"""Project-composed CLI for the Amazon dataset project."""

from __future__ import annotations

from broadway.cli import main
from project import activate_config_overlay

activate_config_overlay()


if __name__ == "__main__":
    main()
