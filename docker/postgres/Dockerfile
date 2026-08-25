# Build source for the broadway-postgres image demanded by
# k8s/postgres-deployment.yaml and docker-compose.yml (`build: ./docker/postgres`
# previously pointed at a context with no Dockerfile).
# Base tag = the judged cluster image (k8s/optuna/postgres.yaml: postgres:16-alpine).
FROM postgres:16-alpine

COPY init.sql /docker-entrypoint-initdb.d/
