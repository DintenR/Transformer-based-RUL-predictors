import pandas as pd
import numpy as np

class ScaniaDataNormalizer:
    def __init__(self):
        self.normatization_params = {}

    def fit(self, x, cluster_col='spec_cluster'):
        self.num_spec = x[cluster_col].nunique()
        self.cluster_col = cluster_col
        epsilon = 1e-6
        for spec in range(self.num_spec):
            spec_indexes = x[x[self.cluster_col] == spec].index.tolist()
            spec_data = x.iloc[spec_indexes]
            spec_x_mean = spec_data.drop(columns=[cluster_col]).mean()
            spec_x_std = np.sqrt(spec_data.drop(columns=[cluster_col]).var()+epsilon)
            self.normatization_params[spec] = {'mean':spec_x_mean,'std':spec_x_std}
    
    def transform(self,x):
        '''
        Normalize the data based on the fitted parameters
        x: pandas dataframe including cluster column
        Returns: pandas dataframe with normilized data based on the fitted parameters
        does not return the cluster column.
        '''
        x_transformed = pd.DataFrame()
        for spec in range(self.num_spec):
            spec_indexes = x[x[self.cluster_col] == spec].index.tolist()
            spec_data = x.iloc[spec_indexes]
            spec_mean = (self.normatization_params[spec])['mean']
            spec_std = (self.normatization_params[spec])['std']
            spec_data = (spec_data.drop(columns=[self.cluster_col]) - spec_mean) / spec_std
            spec_data[self.cluster_col] = spec
            x_transformed = pd.concat([x_transformed,spec_data]).copy()
        return x_transformed.drop(columns=[self.cluster_col])
    
    def fit_transform(self,x):
        self.fit(x)
        return self.transform(x)
    

class ZScoreNormalizer:
    def __init__(self):
        self.normatization_params = {}

    def fit(self, x):
        epsilon = 1e-6
        self.mean = x.mean()
        self.std = np.sqrt(x.var()+epsilon)
    
    def transform(self,x):
        '''
        Normalize the data based on the fitted parameters
        x: pandas dataframe
        Returns: pandas dataframe with normilized data based on the fitted parameters
        '''
        x_transformed = (x - self.mean) / self.std
        return x_transformed
    
    def fit_transform(self,x):
        self.fit(x)
        return self.transform(x)
    

class HistogramFeatureNormalizer:
    def __init__(self):
        self.normatization_params = {}
        self.histogram_features = {}
        bins_per_feature = {
            '167':10,
            '272':10,
            '291':11,
            '158':10,
            '459':20,
            '397':36,
        }
        for feat,bins in bins_per_feature.items():
            self.histogram_features[feat] = [f'{feat}_{bin}' for bin in range(bins)]
    def fit(self, x, bins=100):
        epsilon = 1e-6
        for feature, columns in self.histogram_features.items():
            # get average sum of values for the feature
            feature_sum = x[columns].sum(axis=1).mean() 
            self.normatization_params[feature] = feature_sum + epsilon

    def transform(self,x):
        '''
        Normalize the data based on the fitted parameters
        x: pandas dataframe
        Returns: pandas dataframe with normilized data based on the fitted parameters
        '''
        x_transformed = x.copy()
        for feature, columns in self.histogram_features.items():
            feature_sum = self.normatization_params[feature]
            x_transformed[columns] = x[columns] / feature_sum
        return x_transformed
    
    def fit_transform(self,x):
        self.fit(x)
        return self.transform(x)

    