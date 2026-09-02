# Build source for the broadway-postgres image used by
# k8s/postgres-deployment.yaml and docker-compose.yml.
FROM postgres:16-alpine

COPY init.sql /docker-entrypoint-initdb.d/
