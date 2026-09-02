# Build source for the broadway-postgres image used by
# k8s/postgres-deployment.yaml and docker-compose.yml.
FROM postgres:16-alpine

# SSOT: project/config/layout.yaml build.copy_files (docker/postgres/init.sql)
COPY init.sql /docker-entrypoint-initdb.d/
