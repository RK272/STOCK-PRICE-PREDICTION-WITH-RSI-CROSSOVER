from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    #valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_object2_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_train_file_path1: str

@dataclass
class regressionMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float

@dataclass
class regressionMetricArtifact1:
    mse: float
    rmse: float
    mae: float
    r2: float

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: regressionMetricArtifact
    test_metric_artifact: regressionMetricArtifact


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    train_model_metric_artifact: regressionMetricArtifact
    best_model_metric_artifact: regressionMetricArtifact

@dataclass
class ModelPusherArtifact:
    saved_model_path:str
    model_file_path:str