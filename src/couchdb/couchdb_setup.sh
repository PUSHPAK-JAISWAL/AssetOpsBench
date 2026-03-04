#!/bin/sh -xe

cat >/opt/couchdb/etc/local.ini <<EOF
[couchdb]
single_node=true

[admins]
${COUCHDB_USERNAME} = ${COUCHDB_PASSWORD}
EOF

echo "Starting CouchDB..."
exec /opt/couchdb/bin/couchdb
