import sys
from typing import Optional
import numpy as np
import pandas as pd
import json
from src.configuration.casandradb_connection import CassandraClient
from src.constant.database import keyspace_name,secure_connect_bundle_path,token_file_path,table_name
from src.exception import SensorException

class stockdata:


    """this class help to entire mongo db record as pandas dataframe"""
    def __init__(self):


        try:
            #initialising casandra connection

            self.casandra_client=CassandraClient(keyspace_name=keyspace_name, secure_connect_bundle_path=secure_connect_bundle_path, token_file_path=token_file_path)
        except Exception as e:
            raise SensorException(e,sys)

    #this function exporting data from database to csv
    def export_collection_as_dataframe(

        self,database_name:Optional[str]=None) -> pd.DataFrame:


        try:
            """export entire collection as dataframe from database"""

            session =self.casandra_client.session
            query = f"SELECT * FROM {keyspace_name}.{table_name};"
            result = session.execute(query)
            df = pd.DataFrame(list(result))


            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df

        except Exception as e:
            raise SensorException(e,sys)


