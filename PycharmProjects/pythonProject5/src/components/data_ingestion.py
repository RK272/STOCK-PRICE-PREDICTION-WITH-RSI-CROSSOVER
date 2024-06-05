from src.exception import  SensorException
from src.logger import logging
import os
import sys
from pandas import DataFrame
from src.dataaccess.stock_data import stockdata
from src.entity.config_entity import DataIngestionConfig,DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
            #self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
            #self.fyersdataconfig=fyersdataconfig()

        except Exception as e:
            raise SensorException(e,sys)





    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Exporting data from casandra to feature store ")
            stock_data=stockdata()

            dataframe= stock_data.export_collection_as_dataframe()
            print(dataframe)
            #artifact - dataingestion
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info("artifact-datainjection folder created ")
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise SensorException(e,sys)

    def split_data_as_train_test(self,dataframe:DataFrame) -> None:
        try:
            #train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("performed train test split on dataframe")
            logging.info("exited split_data as train test split")
            dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            #dir_path1 = os.path.dirname(self.data_ingestion_config.testing_file_path)
            #os.makedirs(dir_path1, exist_ok=True)
            logging.info(f"exporting train and test file path")
            dataframe.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            #test_set.to_csv(
               # self.data_ingestion_config.testing_file_path,index=False, header=True
            #)
            logging.info(f"exported train and test file path")
        except Exception as e:
            raise SensorException(e,sys)


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe=self.export_data_into_feature_store()

            self.split_data_as_train_test(dataframe=dataframe)
            """HERE BELOW ONE IS CORRECT DATAINGECTION ARTIFACT I COMMENTED THAT BCZ MY CASSANDRA HAVING DATA UPDATE PROBLEM"""
            #data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path)
            #test_file_path=self.data_ingestion_config.testing_file_path)
            data_ingestion_artifact=DataIngestionArtifact(trained_file_path='artifact/1RELIANCE.csv')
            return data_ingestion_artifact

        except Exception as e:

            raise SensorException(e, sys)


