import sys 
import os 
from src.exception import CustomException 
from src.logger import logging 
from src.utils import load_obj
import pandas as pd

class PredictPipeline: 
    def __init__(self) -> None:
        pass

    def predict(self, features): 
        try: 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e: 
            logging.info("Error occured in predict function in prediction_pipeline location")
            raise CustomException(e,sys)
        
class CustomData: 
        def __init__(self, Age:float, 
                     Squat1Kg:float, 
                     Bench1Kg:float, 
                     Deadlift1Kg:float, 
                     TotalKg:float,
                     Wilks:float,
                     McCulloch:float,
                     Glossbrenner:float,
                     IPFPoints:float, 
                     Sex:str, 
                     Event:str, 
                     Equipment:str,
                     Place:str,
                     Country:str,
                     MeetCountry:str,
                     MeetState:str): 
             self.Age = Age
             self.Squat1Kg = Squat1Kg
             self.Bench1Kg = Bench1Kg
             self.Deadlift1Kg = Deadlift1Kg
             self.TotalKg = TotalKg
             self.Wilks = Wilks 
             self.McCulloch= McCulloch
             self.Glossbrenne = Glossbrenner
             self.IPFPoints = IPFPoints 
             self.Sex = Sex
             self.Event = Event 
             self.Equipment = Equipment
             self.Place = Place
             self.Country = Country
             self. MeetCountry = MeetCountry
             self.MeetState = MeetState
        
        def get_data_as_dataframe(self): 
             try: 
                  custom_data_input_dict = {
                       'Sex': [self.Sex], 
                       'Event': [self.Event], 
                       'Equipment': [self.Equipment], 
                       'Place ': [self.Place],
                       'Country':[self.Country],
                       'MeetCountry':[self.MeetCountry],
                       'MeetState':[self.MeetState],  
                       'Squat1Kg': [self.Squat1Kg], 
                       'Bench1Kg': [self.Bench1Kg], 
                       'Deadlift1Kg': [self.Deadlift1Kg],
                       'TotalKg': [self.TotalKg],
                       'Wilks': [self.Wilks],
                       'McCulloch': [self.McCulloch],
                       'Glossbrenne': [self.Glossbrenne],
                       'IPFPoints': [self.IPFPoints],
                  }
                  df = pd.DataFrame(custom_data_input_dict)
                  logging.info("Dataframe created")
                  return df
             except Exception as e:
                  logging.info("Error occured in get_data_as_dataframe function in prediction_pipeline")
                  raise CustomException(e,sys) 
             
             
        