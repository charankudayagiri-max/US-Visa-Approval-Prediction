import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from typing import Tuple

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from us_visa.entity.estimator import TargetValueMapping

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        :param data_validation_artifact: Output reference of data validation artifact stage
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            
            # Validate schema configuration
            self._validate_schema_config()
            
        except Exception as e:
            raise USvisaException(e, sys)

    def _validate_schema_config(self):
        """Validate that all required schema configurations are present"""
        required_keys = ['oh_columns', 'or_columns', 'transform_columns', 'num_features', 'drop_columns']
        for key in required_keys:
            if key not in self._schema_config:
                raise USvisaException(f"Missing required schema configuration: {key}", sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def _create_company_age_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create company_age feature"""
        try:
            if 'yr_of_estab' not in df.columns:
                raise USvisaException("'yr_of_estab' column not found in dataframe", sys)
            
            df_copy = df.copy()
            df_copy['company_age'] = CURRENT_YEAR - df_copy['yr_of_estab']
            
            # Handle potential negative values or outliers
            df_copy['company_age'] = df_copy['company_age'].clip(lower=0)
            
            logging.info("Successfully created company_age feature")
            return df_copy
            
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Get column lists from schema config
            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            logging.info("Initializing preprocessing transformers")

            # Create transformers
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
            ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            
            # Power transformer for skewed features
            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ],
                remainder='passthrough',  # Keep columns not specified in any transformer
                n_jobs=-1  # Use all available cores
            )

            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            
            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys) from e

    def _prepare_features_and_target(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables"""
        try:
            # Create copy to avoid modifying original data
            df_processed = df.copy()
            
            # Create company_age feature
            df_processed = self._create_company_age_feature(df_processed)
            
            # Drop specified columns
            drop_cols = self._schema_config['drop_columns']
            df_processed = drop_columns(df=df_processed, cols=drop_cols)
            
            # Separate features and target
            if TARGET_COLUMN not in df_processed.columns:
                raise USvisaException(f"Target column '{TARGET_COLUMN}' not found in dataframe", sys)
                
            input_features = df_processed.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature = df_processed[TARGET_COLUMN]
            
            # Map target values
            target_mapping = TargetValueMapping()._asdict()
            target_feature = target_feature.map(target_mapping)
            
            # Validate target mapping
            if target_feature.isnull().any():
                raise USvisaException("Invalid target values found after mapping", sys)
            
            logging.info(f"Successfully prepared features and target for {'training' if is_training else 'testing'} dataset")
            return input_features, target_feature
            
        except Exception as e:
            raise USvisaException(e, sys)

    def _apply_smoteenn(self, features: np.ndarray, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTEENN for handling class imbalance"""
        try:
            logging.info("Applying SMOTEENN for class imbalance handling")
            
            smt = SMOTEENN(
                sampling_strategy="minority",
                random_state=42  # For reproducibility
            )
            
            features_resampled, target_resampled = smt.fit_resample(features, target)
            
            logging.info(f"Original dataset shape: {features.shape}")
            logging.info(f"Resampled dataset shape: {features_resampled.shape}")
            
            return features_resampled, target_resampled
            
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(f"Data validation failed: {self.data_validation_artifact.message}")

            logging.info("Starting data transformation")
            
            # Get preprocessor
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            # Read data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            logging.info(f"Training data shape: {train_df.shape}")
            logging.info(f"Testing data shape: {test_df.shape}")

            # Prepare features and target
            input_feature_train_df, target_feature_train_df = self._prepare_features_and_target(train_df, is_training=True)
            input_feature_test_df, target_feature_test_df = self._prepare_features_and_target(test_df, is_training=False)

            # Apply preprocessing
            logging.info("Applying preprocessing object on training and testing dataframes")
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Successfully applied preprocessing transformations")

            # Handle class imbalance
            input_feature_train_final, target_feature_train_final = self._apply_smoteenn(
                input_feature_train_arr, target_feature_train_df
            )
            
            input_feature_test_final, target_feature_test_final = self._apply_smoteenn(
                input_feature_test_arr, target_feature_test_df
            )

            # Create final arrays
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            # Save artifacts
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Saved all data transformation artifacts")

            # Create and return artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            
            logging.info("Data transformation completed successfully")
            return data_transformation_artifact

        except Exception as e:
            logging.error("Error during data transformation")
            raise USvisaException(e, sys) from e