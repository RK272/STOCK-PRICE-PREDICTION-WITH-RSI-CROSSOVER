import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from src.constant.training_pipeline import TARGET_COLUMN
import pandas_ta as ta
from src.utils.main_utils import read_yaml_file
from src.components.datatransformation1 import NameDropper,rsi,generate_rsi_crosover_sucsess,generate_rsi_crosover_sucsess1
from sklearn.model_selection import train_test_split
from src.logger import logging
import datetime
from datetime import datetime
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import DataTransformationConfig
from src.exception import SensorException
from src.logger import logging
from src.ml.model.estimator import TargetValueMapping
from src.utils.main_utils import save_numpy_array_data, save_object
from src.constant.training_pipeline import SCHEMA_FILE_PATH

class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact,
                    data_transformation_config: DataTransformationConfig):
        """

        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise SensorException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e, sys)




    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:

            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            dropper = NameDropper()
            rsisma= rsi()
            generate_rsi_crosover_sucsess12= generate_rsi_crosover_sucsess()

            # data=self.get_rsi_and_sma()
            preprocessor = Pipeline(
                steps=[
                    ("rsical", rsisma),
                    ("cros", generate_rsi_crosover_sucsess12),
                    #("namedrop",dropper),
                    #("Imputer", simple_imputer),
                   # ("RobustScaler", robust_scaler)


                    # keep every feature in same range and handle outlier
                ]
            )

            return preprocessor

        except Exception as e:
            raise SensorException(e, sys) from e

    @classmethod
    def get_data_transformer_object1(cls) -> Pipeline:
        try:

            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            dropper = NameDropper()
            rsisma = rsi()
            generate_rsi_crosover_sucsess122 = generate_rsi_crosover_sucsess1()

            # data=self.get_rsi_and_sma()
            preprocessor = Pipeline(
                steps=[
                    ("rsical", rsisma),
                    ("cros", generate_rsi_crosover_sucsess122),
                    # ("namedrop",dropper),
                    # ("Imputer", simple_imputer),
                    # ("RobustScaler", robust_scaler)

                    # keep every feature in same range and handle outlier
                ]
            )

            return preprocessor

        except Exception as e:
            raise SensorException(e, sys) from e


    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        try:
            logging.info("iniitiating data transformation")

            train_df=pd.read_csv(self.data_validation_artifact.valid_train_file_path)




            preprocessor = self.get_data_transformer_object()

            preprocessor1 = self.get_data_transformer_object1()
            logging.info("fitting data to preprocessor object ")
            preprocessor.fit(train_df)
            preprocessor1.fit(train_df)

            result_df3 = preprocessor.transform(train_df)

            result_df33= preprocessor1.transform(train_df)


            #date_format = "%Y-%m-%d %H:%M:%S"
            date_format = "%d-%m-%Y %H:%M"
            result_df33['NumericTimestamp1'] = result_df33['DateTime'].apply(lambda x: datetime.strptime(x, date_format).timestamp())

            result_df3['NumericTimestamp'] = result_df3['DateTime'].apply(
                lambda x: datetime.strptime(x, date_format).timestamp())
            df3=result_df3
            df33=result_df33
            matching_rows_large = df33[df33['NumericTimestamp1'].isin(df3['NumericTimestamp'])]

            # Get the remaining rows from the larger DataFrame
            df33 = df33[~df33['NumericTimestamp1'].isin(df3['NumericTimestamp'])]
            df33=df33.reset_index(drop=True)
            num_rows = len(df33)
            new_column_name = 'result'

            # Set the first 4272 rows to 0, leaving the rest unchanged
            df33[new_column_name] = 0
            df33.loc[:num_rows - 1, new_column_name] = 0
            num_rows1 = len(df3)
            new_column_name = 'result'

            # Set the first 4272 rows to 0, leaving the rest unchanged
            df3[new_column_name] = 1
            df3.loc[:num_rows - 1, new_column_name] = 1
            combined_df = pd.concat([df3, df33], ignore_index=True)
            train_df=combined_df.drop(columns=['DateTime','NumericTimestamp','NumericTimestamp1'],axis=1)
            train_df=train_df.sample(frac=1, random_state=42).reset_index(drop=True)
            logging.info("created dataframe with rsi14,rsi9,ema5,ema20 and result column  ")

            gf=train_df

            train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
            logging.info("train test split done  ")


            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            logging.info("droped target column from train_df  ")

            target_feature_train_df = train_df[TARGET_COLUMN]
            logging.info("created target column for train_df  ")
            target_feature_train_df = target_feature_train_df.replace(TargetValueMapping().to_dict())
            logging.info("converted  target_feature_train_df to dictionory ")


            # testing dataframe


            input_feature_test_df= test_df.drop(columns=[TARGET_COLUMN], axis=1)
            logging.info("droped target column from test_df  ")
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("created target column for test_df  ")
            target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().to_dict())
            logging.info("converted  target_feature_test_df to dictionory ")


            train_arr = np.c_[input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]
            logging.info("created train_arr and test_arr")


            # save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr, )
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor, )
            save_object(self.data_transformation_config.transformed_object_file_path2, preprocessor1, )
            train_df.to_csv(self.data_transformation_config.transformed_object_file_path3, index=False, header=True)
            logging.info("save test.np and train.np to transformed folder and preprocesor object to transformed object folder")

            # preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_object2_file_path=self.data_transformation_config.transformed_object_file_path2,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_train_file_path1=self.data_transformation_config.transformed_object_file_path3
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            logging.info("data transformation completed ")
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys) from e

