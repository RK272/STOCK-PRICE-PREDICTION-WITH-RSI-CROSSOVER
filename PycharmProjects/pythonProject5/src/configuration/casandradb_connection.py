from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json
import os
import pandas as pd
import sys
import cassandra

#creating cassandraclient for cassandra connecton

class CassandraClient:
    client = None

    def __init__(self, keyspace_name, secure_connect_bundle_path, token_file_path):
        try:
            if CassandraClient.client is None:
                if not os.path.exists(secure_connect_bundle_path):
                    raise FileNotFoundError(f"Secure connect bundle not found at {secure_connect_bundle_path}")
                if not os.path.exists(token_file_path):
                    raise FileNotFoundError(f"Token file not found at {token_file_path}")

                with open(token_file_path) as f:
                    secrets = json.load(f)

                CLIENT_ID = secrets["clientId"]
                CLIENT_SECRET = secrets["secret"]

                cloud_config = {
                    'secure_connect_bundle': secure_connect_bundle_path
                }

                auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
                CassandraClient.client = Cluster(cloud=cloud_config, auth_provider=auth_provider)

            self.cluster = CassandraClient.client
            self.session = self.cluster.connect()
            self.session.set_keyspace(keyspace_name)
            self.keyspace_name = keyspace_name
        except Exception as e:
            raise e