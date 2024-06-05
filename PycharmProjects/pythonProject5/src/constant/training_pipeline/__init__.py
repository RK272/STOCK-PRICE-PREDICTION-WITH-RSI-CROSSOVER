import os
TARGET_COLUMN='result'

MODEL_FILE_NAME1 = "model1.pkl"
"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
ARTIFACT_DIR:str ="artifact"
DATA_INGESTION_DIR_NAME:str ="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
FILE_NAME: str = "stock.csv"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2
"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
SAVED_MODEL_DIR =os.path.join("saved_models")
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
PREPROCSSING_OBJECT2_FILE_NAME = "preprocessing1.pkl"
PREPROCSSING_OBJECT3_FILE_NAME = "train.csv"

SAVED_MODEL_DIR =os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"

"""
Model Trainer ralated constant start with MODE TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.01
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.7

"""
Model evaluater  ralated constant start with MODE TRAINER VAR NAME
"""
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_NAME= "report.yaml"

MODEL_PUSHER_DIR_NAME = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = SAVED_MODEL_DIR

