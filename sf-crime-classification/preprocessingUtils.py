#Utility for Feature Engineering

from timeit import default_timer as timer
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

CATEGORICAL_COLUMNS = [
    "DayOfWeek",
    "PdDistrict",
    "Address_split_0",
    "Address_split_1"
]

NUMERICAL_FEATURE_COLUMNS = [
    'X',
    'Y',
    'Month',
    'Hour_x',
    'Hour_y',
    'X_radians',
    'Y_radians'
]

GENERATED_CATEGORY_COLUMS = []

def generateDateColumns(df):
    df["Dates"] = pd.to_datetime(df['Dates'])
    df['Date'] = df.Dates.dt.date
    df['Hour'] = df.Dates.dt.hour
    df['Minute'] = df.Dates.dt.minute
    df["Month"] = df.Dates.dt.month
    df["Year"] = df.Dates.dt.year

    return df

def getIndexForYear(df, yearList):
    conditionArray = df['Year'] == -1
    for year in yearList:
        conditionArray = (conditionArray) | (df['Year'] == year)
    
    return df[conditionArray].index[0], df[conditionArray].index[-1] 

def convertCoordinatesToRadians(df):
    df['X_radians'] = df.X.apply(np.radians)
    df['Y_radians'] = df.Y.apply(np.radians)

    return df

def reduceDescriptCardianlity(df):
    descript_dict = df["Descript"].value_counts().to_dict()
    newFeatureList = []
    
    for column in descript_dict.keys():
        if descript_dict[column]>500:
            newFeatureList.append(column)
    df["New_Descript"] = df['Descript'].map(lambda descript: descript if descript in newFeatureList else "Others")

    return df

def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    
    dlat = (lat2 - lat1)
    dlon = (lon2 - lon1)
    
    term1 = (dlat/2.0).apply(np.sin)
    term2 = lat1.apply(np.cos)
    term3 = np.cos(lat2)
    term4 = (dlon/2).apply(np.sin)

    a = (term1 ** 2) + term2 * term3*(term4**2)
    c = a.apply(lambda x: 2*np.arctan2(x**0.5, (1-x)**0.5))
    
    distance  = earth_radius * c
    return c

def findRecentCrimesInVicinty(df_ref, x, y, date, index, radius=0.01, num_days=10):
    if df_ref.shape[0] > 10000:
        df = df_ref.loc[index-1000:index+8000]
    else:
        df = df_ref
    
    day_difference = (date - df['Dates']).dt.days
    day_condition = (day_difference < num_days) & (day_difference >= 0)

    radius_df = haversine(df[day_condition]['X_radians'], df[day_condition]['Y_radians'], np.radians(x), np.radians(y))
    radius_condition = (radius_df <= radius) & (radius_df >= 0) 
    
    df_eligible_frauds = df[day_condition & radius_condition]

    return df_eligible_frauds.shape[0]

def generateCategoryFeatures(df, complete_df):
    for category in df['Category'].value_counts().index:
        start = timer()

        df_ref = complete_df[complete_df['Category'] == category][['X_radians', 'Y_radians', 'Dates']]
        df[category+'_count'] = df[['X', 'Y', 'Dates']].apply(lambda row: findRecentCrimesInVicinty(df_ref, row['X'], row['Y'], row['Dates'], row.name), axis=1)
        GENERATED_CATEGORY_COLUMS.append(category+'_count')
        
        end = timer()
        print (category, str(end - start))
    return df

def generateFeaturesForDF(df, complete_df):
    df = generateCategoryFeatures(df, complete_df)
    df = generateAddressFeature(df)
    df = convertTimeToCyclic(df)

    return df


def generateAddressFeature(df):
    pattern = re.compile("([0-9a-zA-Z\s]*[0-9a-zA-Z])?([\s]*)(/|of)([\s]*)([0-9a-zA-Z\s]*)")

    df["Address_split_0"] =  df["Address"].apply(lambda x : pattern.search(x)[1])
    df["Address_split_1"] =  df["Address"].apply(lambda x : pattern.search(x)[5])

    return df

def dropOutlierRows(df):
    df = df[(df['Y'] < 38) & (df['X'] < -122)]
    return df

def convertTimeToCyclic(df):
    df['hourfloat']=df.Hour +df.Minute/60.0
    df['Hour_x'] = np.sin(2.*np.pi*df.hourfloat/24.)
    df['Hour_y']=np.cos(2.*np.pi*df.hourfloat/24.)

    return df

class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)
    
    def fit_transform(self, data_list):
        self.fit(data_list)
        return self.transform(data_list)

def encodeCategoricalColumns(df):
    encoderDictionary = defaultdict(LabelEncoderExt)
    outputClassEncoder = LabelEncoder()

    df = df[CATEGORICAL_COLUMNS].apply(lambda col: encoderDictionary[col.name].fit_transform(col))
    df["Category"] = outputClassEncoder.fit_transform(df["Category"])

    return df, encoderDictionary, outputClassEncoder
	


