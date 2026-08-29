"""Project composition root for the generic experiment dashboard."""

import os

import uvicorn

from project.paths import load_project_paths


def main() -> None:
    """Serve the generic dashboard with this project's resolved paths."""
    paths = load_project_paths()
    os.environ.setdefault("BROADWAY_EXPERIMENTS_ROOT", str(paths.experiments))
    os.environ.setdefault("BROADWAY_OBSERVATIONS_DIR", str(paths.observations))
    from broadway.reports.experiments_dashboard import app

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
