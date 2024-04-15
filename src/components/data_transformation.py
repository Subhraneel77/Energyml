import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_function

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Sex', 'Event', 'Equipment', 'WeightClassKg', 'Place',
       'Country', 'MeetCountry', 'MeetState']
            numerical_cols = ['Age', 'BodyweightKg', 'Squat1Kg', 'Squat2Kg', 'Squat3Kg', 'Squat4Kg',
       'Best3SquatKg', 'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 'Bench4Kg',
       'Best3BenchKg', 'Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg',
       'Deadlift4Kg', 'Best3DeadliftKg', 'TotalKg', 'Wilks', 'McCulloch',
       'Glossbrenner', 'IPFPoints']
            
            # Define the custom ranking for each ordinal variable
            sex_categories = ['F','M']
            equipment_categories = ['Wraps', 'Raw', 'Single-ply', 'Multi-ply', 'Straps']
            weightclasskg_categories = ['60', '56', '110', '75', '82.5', '52', '67.5', '90', '110+', '125',
       '100', '140', '140+', '48', '90+', '44', '105', '74', '93',
       '120+', '120', '63', '83', '84', '57', '72', '84+', '47', '59',
       '66', '53', '125+', '43', '100+', '90.7', '90.7+', '36', '40',
       '46', '49', '75+', '82.5+', '52+', '145', '145+', '72+', '93+',
       '60+', '50', '65', '80', '80+', '67.5+', '105+', '63+', '35',
       '155', '39', '155+', '68', '50.5', '55.5', '58.5', '70', '47.5',
       '136', '136+', '55', '54.4', '58.9', '63.5', '68.9', '78.9',
       '78.9+', '56.7', '83.9', '102', '113.4', '127', '127+', '117.5',
       '101', '103', '67', '82', '36.2', '52.5', '72.5', '81.6', '104.3',
       '117.9', '117.9+', '79.3', '113.4+', '49.9', '57.1', '70.3',
       '77.1', '92.9', '68.5', '69.5', '70.5', '71.5', '73.5', '74.5',
       '75.5', '76.5', '77.5', '78.5', '79.5', '80.5', '81.5', '77', '85',
       '113.5', '62.5', '57.5', '54', '92.5', '143', '61.2', '87.5']
            place_categories = ['4', '2', '1', '3', '5', '7', '6', '9', 'DQ', '8', '11', '12',
       '13', '15', '10', '14', '18', '16', '17', '19', '25', '32', '23',
       '24', '27', '28', '29', '21', '26', '31', '22', '30', '20', '33',
       '34', '36', '37', 'G', '35', '38', 'NS', 'DD']
            country_categories = [ 'Australia', 'New Zealand', 'Cayman Islands', 'USA',
       'Ireland', 'Canada', 'India', 'Czechia', 'Argentina', 'Austria',
       'South Africa', 'England', 'UK', 'Papua New Guinea', 'Germany',
       'Japan', 'Switzerland', 'West Germany', 'Poland', 'Hungary',
       'Belgium', 'Tajikistan', 'Russia', 'Netherlands', 'Armenia',
       'Croatia', 'Slovenia', 'Turkey', 'Slovakia', 'Italy', 'Belarus',
       'Brazil', 'Wales', 'Chile', 'Ukraine', 'Moldova', 'Uzbekistan',
       'Kazakhstan', 'Latvia', 'Spain', 'Georgia', 'Israel', 'Azerbaijan',
       'Estonia', 'Luxembourg', 'Mexico', 'Lithuania', 'Iceland', 'Egypt',
       'Nigeria', 'Bulgaria', 'Puerto Rico', 'Bahamas', 'Costa Rica',
       'Uruguay', 'Peru', 'US Virgin Islands', 'China', 'Cook Islands',
       'Finland', 'Kyrgyzstan', 'Portugal', 'Scotland', 'Indonesia',
       'Sweden', 'France', 'Norway', 'Taiwan', 'N.Ireland', 'Guatemala',
       'Philippines', 'Nauru', 'Guyana', 'Tonga', 'Tahiti', 'Fiji',
       'New Caledonia', 'Denmark', 'Colombia', 'Ecuador',
       'Trinidad and Tobago', 'Jamaica', 'USSR', 'Cyprus', 'Pakistan']
            meetcountry_categories = ['Australia', 'Germany', 'Croatia', 'USA', 'South Africa', 'Chile',
       'Ukraine', 'Austria', 'Mexico', 'Ireland', 'Netherlands', 'Wales',
       'Japan', 'Slovenia', 'Kazakhstan', 'Poland', 'New Zealand',
       'Italy', 'UK', 'England', 'Hungary', 'Canada', 'Scotland',
       'Lithuania', 'Indonesia', 'Tahiti', 'Denmark', 'USSR', 'France',
       'Sweden', 'Russia', 'Belgium']
            meetstate_categories = ['VIC', 'QLD', 'SA', 'NSW', 'ACT', 'WA', 'NT', 'MV', 'BB',
       'BY', 'NW', 'SN', 'BW', 'TH', 'ST', 'NI', 'HE', 'SH', 'BE', 'RP',
       'WV', 'RI', 'OR', 'IL', 'TX', 'MI', 'CA', 'OH', 'NY', 'FL', 'KS',
       'TN', 'MA', 'NV', 'PA', 'WI', 'AR', 'IN', 'MO', 'AL', 'NH', 'MS',
       'UT', 'DE', 'NJ', 'CO', 'CT', 'VT', 'MD', 'AK', 'GA', 'AKL', 'WKO',
       'LA', 'NTL', 'AZ', 'NC', 'VA', 'OK', 'HI', 'NS', 'MN', 'IA', 'OTA',
       'QC', 'TAS', 'DC', 'HKB', 'ID', 'WGN']
         
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[sex_categories,equipment_categories,weightclasskg_categories,place_categories,country_categories,meetcountry_categories,meetstate_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'BodyweightKg'
            drop_columns = [target_column_name,'Name']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Transformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_function(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)