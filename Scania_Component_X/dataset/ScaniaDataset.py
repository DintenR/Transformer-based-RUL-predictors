import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataset.utils import ScaniaDataNormalizer, ZScoreNormalizer, HistogramFeatureNormalizer
from kmodes.kmodes import KModes


#Pytorch dataset class for Scania dataset
class ScaniaDataset(Dataset):

    # get the subsets of the data
    
    @staticmethod
    def __get_stratified_split(train_data,validation_rate=0.2,undersample=0):
        '''
        Get a stratified split of the data
        train_data: data to split
        validation_rate: percentage of the data to use as validation data
        '''
        # Get vehicle ids
        vehicle_ids = train_data['vehicle_id'].unique()
        vehicle_last_rows = []
        # Get the last row of each vehicle
        for id in vehicle_ids:
            vehicle_last_rows.append(train_data[train_data['vehicle_id']==id].index[-1])
        print(f'Vehicle count: {len(vehicle_last_rows)}')
        # Get the last label of each vehicle
        vehicle_clases = train_data.iloc[vehicle_last_rows,:]
        if undersample > 0:
            print('Undersampling')
            for label in vehicle_clases['class_label'].unique():
                print(f'Class {label} count: {vehicle_clases[vehicle_clases["class_label"]==label].shape[0]}')
            vehicle_clases = pd.concat( [vehicle_clases[vehicle_clases['class_label']==0].sample(frac=undersample),vehicle_clases[vehicle_clases['class_label']!=0]] )
        for label in vehicle_clases['class_label'].unique():
            print(f'Class {label} count: {vehicle_clases[vehicle_clases["class_label"]==label].shape[0]}')
        # Split the vehicle ids into train and validation stratified by class label
        if validation_rate == 0:
            return train_data[train_data['vehicle_id'].isin(vehicle_clases['vehicle_id'])].index, []
        train_ids, val_ids = train_test_split(vehicle_clases['vehicle_id'],test_size=validation_rate,stratify=vehicle_clases['class_label'])
        print(f'Train ids: {len(train_ids)}')
        print(f'Validation ids: {len(val_ids)}')
        # Get the indexes of the events for vehicles in train and validation data
        train_idxs = train_data[train_data['vehicle_id'].isin(train_ids)].index
        print(f'Train indexes: {len(train_idxs)}')
        val_idxs = train_data[train_data['vehicle_id'].isin(val_ids)].index
        print(f'Validation indexes: {len(val_idxs)}')

        print(len(train_data.iloc[train_idxs]))
        print(len(train_data.iloc[val_idxs]))
    
        return train_idxs, val_idxs
    
    @staticmethod
    def __sequence_indexes(subset_first_index, subset_last_index,window_size, only_last_window=False):
        '''
        Compute the indexes of the sequences in the dataset
        x: features
        y: target
        window_size: size of the window in number of observations
        '''
        if only_last_window:
            end_idx = [subset_last_index]
        else:
            end_idx = np.arange(start=subset_first_index,stop=subset_last_index+1,dtype=int)
        
        start_idx = np.vectorize(lambda x: max(subset_first_index,x-window_size))(end_idx)
        return start_idx, end_idx
    
    @staticmethod
    def __dataset_from_subsets(subsets, target,window_size,drop_cols=[],only_last_window=False):
        '''
        Pick a bunch of datasets, perform window tranformation and join all the windowed
        data on a unique dataset
        subset: iterable containing the subsets to join
        target: target variable labels in the subsets
        window_size: size of the window to group the data
        drop_cols: list of column labels that have to be remove form the final dataset
        only_last_window: if True, only the last window of each subset is considered
        return: x, y, sequence_starting_indexes, sequence_ending_indexes
        '''
        subsets.sort_values(['vehicle_id','time_step'],inplace=True)
        subsets.reset_index(drop=True,inplace=True)

        # Get vehicle ids to filter data
        vehicle_ids = subsets['vehicle_id'].unique()

        # Initialize lists to store the indexes of the sequences
        sequence_starting_indexes_list = []
        sequence_ending_indexes_list = []
        
        # for each vehicle in the dataset
        for id in vehicle_ids:
            # get the subset of the data for the vehicle
            subset = subsets[subsets['vehicle_id']==id]

            # get the indexes first and last indexes of the subset
            subset_first_index = subset.index[0]
            subset_last_index = subset.index[-1]
            # print(f'Vehicle {id} first index: {subset_first_index} last index: {subset_last_index}')

            # get the statring and ending indexes of the sequences for the subset
            start_idx, end_idx = ScaniaDataset.__sequence_indexes(subset_first_index,subset_last_index,window_size,only_last_window) # get the indexes of the sequences for one subset

            # append the indexes to the lists
            sequence_starting_indexes_list.extend(start_idx)
            sequence_ending_indexes_list.extend(end_idx)
                        
        # print("All sequences concatenated")
        # print('Total sequences:',len(sequence_starting_indexes_list))

        return np.array(subsets.drop(columns=[target,*drop_cols],errors='ignore')),np.array(subsets[target]),  sequence_starting_indexes_list, sequence_ending_indexes_list

    def __one_hot_encode(self, data, num_classes):
        '''
        One hot encode the data
        data: data to encode
        num_classes: number of classes to encode
        '''
        return np.eye(num_classes)[data]

    def __init__(self, data, labels, columns, seq_start_indexes, seq_end_indexes, window_size, transform=None, only_two_classes=False):
        super().__init__()
        self.data = data
        self.labels = labels
        self.columns = columns 
        if only_two_classes:
            # Convert labels to binary classification problem to try to minimize cost function
            self.labels = np.vectorize(lambda x: 1 if x > 0 else 0)(self.labels)
        else:
            self.labels = self.__one_hot_encode(self.labels,5)
        self.transform = transform
        self.__sequence_start_indexes = seq_start_indexes
        self.__sequence_end_indexes = seq_end_indexes
        self.sequence_end_indexes = seq_end_indexes
        self.__window_size = window_size
        
    @staticmethod
    def get_subsets(data_dir,piece_wise_rul = 0):
        return ScaniaDataset.__get_subsets(data_dir,piece_wise_rul)

    @staticmethod
    def __per_vehicle_null_ffill(data):
        
        data['GROUP'] = data.groupby('vehicle_id').ngroup()
        data.set_index(['GROUP'], inplace=True)
        data.sort_index(inplace=True)
        data = data.ffill() * (1 - data.isnull().astype(int)).groupby(level=0).cumsum().map(lambda x: None if x == 0 else 1)
        data.reset_index(inplace=True, drop=True)
        return data

    @staticmethod
    def __tte_to_label(x):
        label = 0 # base case > 48 or < 0
        if x > 0 and x <= 6:
            label = 4
        elif x > 6 and x <= 12:
            label = 3
        elif x > 12 and x <= 24:
            label = 2
        elif x > 24 and x <= 48:
            label = 1
        return label
         
    @staticmethod
    def __get_subsets(data_dir, piecewise_rul=0, validation_rate=0.2, cluster_specifications=False,undersample=0,include_specifications=True,histogram_normalizer=False,foward_fill=True):
        '''
        Load data from Scania dataset files and return a list of subsets
        data_dir: directory where the dataset is stored
        piecewise_rul: maximum RUL value for each experiment
        validation_rate: percentage of the data to use as validation data
        cluster_specifications: if True, cluster vehicle specifications and include cluster labels as features
        undersample: if > 0, undersample the majority class by the given factor
        include_specifications: if True, include vehicle specifications as features
        histogram_normalizer: if True, normalize histogram features using custom normalizer
        foward_fill: if True, apply forward fill to fill missing values in each vehicle's readouts
        return: training, validation and test subsets
        '''
        # Prepare feature names
        counter_features = ['171_0','666_0','427_0','837_0','309_0','835_0','370_0','100_0']
        histogram_features = []
        bins_per_feature = {
            '167':10,
            '272':10,
            '291':11,
            '158':10,
            '459':20,
            '397':36,
        }
        for feat,bins in bins_per_feature.items():
            histogram_features.extend([f'{feat}_{bin}' for bin in range(bins)])

        specification_features = [f'Spec_{cat}' for cat in range(8)]
        specification_cluster_feature = ['spec_cluster']

        # Load train data
        data_dir = data_dir
        train_readouts = pd.read_csv(f'{data_dir}/train_operational_readouts.csv')
        
        # Apply forward fill to fill missing values in each vehicle's readouts
        if foward_fill:
            train_readouts = ScaniaDataset.__per_vehicle_null_ffill(train_readouts)
        # Fill remaining missing values with 0
        train_readouts.fillna(0,inplace=True)

        # Sort data by vehicle_id and time_step
        train_readouts.sort_values(['vehicle_id','time_step'],inplace=True)
        train_readouts.reset_index(drop=True,inplace=True)
        train_tte = pd.read_csv(f'{data_dir}/train_tte.csv')
        train_specifications = pd.read_csv(f'{data_dir}/train_specifications.csv')

        # Compute RUL labels on train
        train_data = pd.merge(train_readouts,train_tte,on='vehicle_id')
        train_data['class_label'] = train_data['length_of_study_time_step']-train_data['time_step']

        # Remove rows with class_label < 48 and did not fail in study
        rows_to_drop = (train_data[(train_data['class_label'] < 48) & (train_data['in_study_repair'] == 0)]).index
        train_data.drop(rows_to_drop, inplace=True)
        train_data.reset_index(drop=True,inplace=True)
        train_data['class_label'] = train_data['class_label'].map(lambda x: ScaniaDataset.__tte_to_label(x))
        train_data.iloc[train_data[(train_data['in_study_repair'] == 0)].index,train_data.columns.get_loc('class_label')] = 0

        # Use original validation and test data 
        if validation_rate == 0:
            # print(f'{train_readouts.shape[0]} Train readouts loaded')
            # print(f'{train_specifications.shape[0]} Train specifications loaded')
            
            # Load validation data
            train_idx, _ = ScaniaDataset.__get_stratified_split(train_data,validation_rate=validation_rate,undersample=undersample)
            train_data = train_data.iloc[train_idx].copy()
            val_readouts = pd.read_csv(f'{data_dir}/validation_operational_readouts.csv')
            # print(f'{val_readouts.shape[0]} Validation readouts loaded')

            # Apply forward fill to fill missing values in each vehicle's readouts
            if foward_fill:
                val_readouts = ScaniaDataset.__per_vehicle_null_ffill(val_readouts)
            # Fill remaining missing values with 0
            val_readouts.fillna(0,inplace=True)

            # Sort data by vehicle_id and time_step
            val_readouts.sort_values(['vehicle_id','time_step'],inplace=True)
            val_specifications = pd.read_csv(f'{data_dir}/validation_specifications.csv')

            # print(f'{val_specifications.shape[0]} Validation specifications loaded')
            val_labels = pd.read_csv(f'{data_dir}/validation_labels.csv')
            val_data = pd.merge(val_readouts,val_labels,on='vehicle_id')
            
            # Load test data
            test_data = pd.read_csv(f'{data_dir}/test_operational_readouts.csv')
            # print(f'{test_data.shape[0]} Test readouts loaded')

            # Apply forward fill to fill missing values in each vehicle's readouts
            if foward_fill:
                test_data = ScaniaDataset.__per_vehicle_null_ffill(test_data)
            # Fill remaining missing values with 0
            test_data.fillna(0,inplace=True)

            # Sort data by vehicle_id and time_step
            test_data.sort_values(['vehicle_id','time_step'],inplace=True)
            test_data.reset_index(drop=True,inplace=True)
            test_data['class_label'] = 0 # test data has no class labels
            test_specifications = pd.read_csv(f'{data_dir}/test_specifications.csv')
            # print(f'{test_specifications.shape[0]} Test specifications loaded')
        else:
            # Split train for validation and use validation data as test data
            train_idx, val_idx = ScaniaDataset.__get_stratified_split(train_data,validation_rate=validation_rate,undersample=undersample)
            val_data = train_data.iloc[val_idx].copy().reset_index(drop=True)
            train_data = train_data.iloc[train_idx].copy()
            val_specifications = train_specifications[train_specifications['vehicle_id'].isin(val_data['vehicle_id'])].copy().reset_index(drop=True)
            train_specifications = train_specifications[train_specifications['vehicle_id'].isin(train_data['vehicle_id'])].copy()
            # print(f'{train_data.shape[0]} Train readouts loaded')
            # print(f'{train_specifications.shape[0]} Train specifications loaded')
            # print(f'{val_data.shape[0]} Validation readouts loaded')
            # print(f'{val_specifications.shape[0]} Validation specifications loaded')
            
            # Load test data
            test_readouts = pd.read_csv(f'{data_dir}/validation_operational_readouts.csv')
            # Apply forward fill to fill missing values in each vehicle's readouts
            if foward_fill:
                test_readouts = ScaniaDataset.__per_vehicle_null_ffill(test_readouts)
            # Fill remaining missing values with 0
            test_readouts.fillna(0,inplace=True)

            test_specifications = pd.read_csv(f'{data_dir}/validation_specifications.csv')
            # print(f'{test_specifications.shape[0]} Validation specifications loaded')
            test_labels = pd.read_csv(f'{data_dir}/validation_labels.csv')
            test_data = pd.merge(test_readouts,test_labels,on='vehicle_id',how='inner')

        # print('Train label counts')
        # print(train_data['class_label'].value_counts())
        # print('Validation label counts')
        # print(val_data['class_label'].value_counts())
        # print('Test label counts')
        # print(test_data['class_label'].value_counts())
        # Replace specification features with cluster labels
        # Compute specification clusters
        
        # Include vehicle specifications as features
        if include_specifications:
            # Cluster vehicle specifications and include cluster labels as features
            if cluster_specifications:
                kmodes = KModes(n_clusters=35, init='Cao', n_init=5, verbose=1)
                kmodes.fit(train_specifications[specification_features])
                train_specifications['spec_cluster'] = kmodes.predict(train_specifications[specification_features])
                val_specifications['spec_cluster'] = kmodes.predict(val_specifications[specification_features])
                test_specifications['spec_cluster'] = kmodes.predict(test_specifications[specification_features])

                train_data = pd.merge(train_data,train_specifications[['vehicle_id','spec_cluster']],on='vehicle_id')
                val_data = pd.merge(val_data,val_specifications[['vehicle_id','spec_cluster']], on='vehicle_id')
                test_data = pd.merge(test_data,test_specifications[['vehicle_id','spec_cluster']],on='vehicle_id')

                # Z-score normalization of histogram and counter features for each vehicle cluster
                normalizer = ScaniaDataNormalizer()
                normalizer.fit(train_data[histogram_features+counter_features+specification_cluster_feature])
                train_data[histogram_features+counter_features] = normalizer.transform(train_data[histogram_features+counter_features+specification_cluster_feature])
                val_data[histogram_features+counter_features] = normalizer.transform(val_data[histogram_features+counter_features+specification_cluster_feature])
                test_data[histogram_features+counter_features] = normalizer.transform(test_data[histogram_features+counter_features+specification_cluster_feature])
                specification_features = ['spec_cluster']
            else:
                # Include vehicle specifications as features without clustering
                train_data = pd.merge(train_data,train_specifications,on='vehicle_id')
                val_data = pd.merge(val_data,val_specifications,on='vehicle_id')
                test_data = pd.merge(test_data,test_specifications,on='vehicle_id')
                for feature in specification_features:
                    train_data[feature] = train_specifications[feature].apply(lambda x: x[3:]).astype(int)
                    val_data[feature] = val_specifications[feature].apply(lambda x: x[3:]).astype(int)
                    test_data[feature] = test_specifications[feature].apply(lambda x: x[3:]).astype(int)
        else:
            # Do not include vehicle specifications as features
            specification_features = []

        # Normalize histogram and counter features separately
        if histogram_normalizer:
            # Normalize histogram features using custom normalizer
            h_normalizer = HistogramFeatureNormalizer()
            h_normalizer.fit(train_data[histogram_features])
            train_data[histogram_features] = h_normalizer.transform(train_data[histogram_features])
            val_data[histogram_features] = h_normalizer.transform(val_data[histogram_features])
            test_data[histogram_features] = h_normalizer.transform(test_data[histogram_features])

            # Normalize counter features using Z-score normalization
            z_normalizer = ZScoreNormalizer()
            z_normalizer.fit(train_data[counter_features + specification_features])
            train_data[counter_features + specification_features] = z_normalizer.transform(train_data[counter_features + specification_features])
            val_data[counter_features + specification_features] = z_normalizer.transform(val_data[counter_features + specification_features])
            test_data[counter_features + specification_features] = z_normalizer.transform(test_data[counter_features + specification_features])
        else:
            # Normalize histogram and counter features using Z-score normalization 
            normalizer = ZScoreNormalizer()
            normalizer.fit(train_data[histogram_features+counter_features+specification_features])
            train_data[histogram_features+counter_features+specification_features] = normalizer.transform(train_data[histogram_features+counter_features+specification_features])
            val_data[histogram_features+counter_features+specification_features] = normalizer.transform(val_data[histogram_features+counter_features+specification_features])
            test_data[histogram_features+counter_features+specification_features] = normalizer.transform(test_data[histogram_features+counter_features+specification_features])


        return train_data, val_data, test_data

    @staticmethod        
    def get_datasets(data_dir, test_pct=0.2, piece_wise_rul = 0, window_size = 30, validation_rate=0,stored_subsets=False,cluster_specifications=False,undersample=0,
                     only_two_classes=False,include_specifications=True,histogram_normalizer=False, forward_fill=True, pca=False):
        '''
        Load the MetroPT dataset and split it into train and test datasets
        data_dir: directory where the dataset is stored
        test_pct: percentage of the data to be used as test data
        piece_wise_rul: maximum RUL value for each experiment
        window_size: size of the window to group the data
        validation_rate: percentage of the data to use as validation data
        stored_subsets: if True, load stored subsets from files
        cluster_specifications: if True, cluster vehicle specifications and include cluster labels as features
        undersample: if > 0, undersample the majority class by the given factor
        only_two_classes: if True, convert labels to binary classification problem
        include_specifications: if True, include vehicle specifications as features
        histogram_normalizer: if True, normalize histogram features using custom normalizer
        forward_fill: if True, apply forward fill to fill missing values in each vehicle's readouts
        pca: if True, apply PCA to reduce the number of features

        returns: training, validation and test datasets
        '''
        if stored_subsets:
            train_subsets = pd.read_csv(f'{data_dir}/train_subsets.csv')
            val_subsets = pd.read_csv(f'{data_dir}/val_subsets.csv')
            test_subsets = pd.read_csv(f'{data_dir}/test_subsets.csv')
        else:
            train_subsets, val_subsets, test_subsets = ScaniaDataset.__get_subsets(data_dir,piece_wise_rul,validation_rate=validation_rate,cluster_specifications=cluster_specifications,
                                                                                   undersample=undersample,include_specifications=include_specifications,
                                                                                   histogram_normalizer=histogram_normalizer,foward_fill=forward_fill)
        drop_cols_train = ['vehicle_id','in_study_repair','length_of_study_time_step'] # including time_step
        if cluster_specifications:
            drop_cols_train.extend(['835_0','427_0'])
        # print(f'{len(train_subsets)} Train subsets loaded')
        # print(f'{len(val_subsets)} Validation subsets loaded')
        # print(f'{len(test_subsets)} Test subsets loaded')
        # print(train_subsets['class_label'].unique())
        # print(val_subsets['class_label'].unique())
        # print(test_subsets['class_label'].unique())

        only_last_window = False if validation_rate > 0 else True
        columns = train_subsets.drop(columns=['class_label',*drop_cols_train]).columns
        x_train, y_train, seq_start_indexes_train, seq_end_indexes_train = ScaniaDataset.__dataset_from_subsets(train_subsets, drop_cols=drop_cols_train, target='class_label',window_size=window_size,only_last_window=False)
        x_val, y_val, seq_start_indexes_val, seq_end_indexes_val = ScaniaDataset.__dataset_from_subsets(val_subsets,drop_cols=drop_cols_train, target='class_label',window_size=window_size,only_last_window=only_last_window)
        x_test, y_test, seq_start_indexes_test, seq_end_indexes_test = ScaniaDataset.__dataset_from_subsets(test_subsets,drop_cols=drop_cols_train, target='class_label',window_size=window_size,only_last_window=True)
    
        if pca:
            pca = PCA(n_components=20)
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_val = pca.transform(x_val)
            x_test = pca.transform(x_test)
        
        train_dataset = ScaniaDataset(x_train, y_train,columns, seq_start_indexes_train, seq_end_indexes_train,window_size=window_size,only_two_classes=only_two_classes)
        val_dataset = ScaniaDataset(x_val, y_val, columns,seq_start_indexes_val, seq_end_indexes_val,window_size=window_size,only_two_classes=only_two_classes)
        test_dataset = ScaniaDataset(x_test, y_test, columns, seq_start_indexes_test, seq_end_indexes_test,window_size=window_size)

        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def create_and_store_subsets(data_dir,validation_rate=0):
        train_subsets, val_subsets, test_subsets = ScaniaDataset.__get_subsets(data_dir,validation_rate=validation_rate)
        train_subsets.to_csv(f'{data_dir}/train_subsets.csv',index=False)
        val_subsets.to_csv(f'{data_dir}/val_subsets.csv',index=False)
        test_subsets.to_csv(f'{data_dir}/test_subsets.csv',index=False)
        


    @staticmethod
    def get_dataloaders(data_dir, test_pct=0.7, piece_wise_rul = 0, window_size = 30, batch_size=32,validation_rate=0.2,stored_subsets=False,cluster_specifications=False,
                        undersample=0,only_two_classes=False,include_specifications=True,histogram_normalizer=False, forward_fill=True,pca=False):
        '''
        Load the MetroPT dataset and split it into train and test datasets
        data_dir: directory where the dataset is stored
        test_pct: percentage of the data to be used as test data
        piece_wise_rul: maximum RUL value for each experiment
        window_size: size of the window to group the data
        batch_size: size of the batch for the dataloaders
        validation_rate: percentage of the data to use as validation data
        stored_subsets: if True, load stored subsets from files
        cluster_specifications: if True, cluster vehicle specifications and include cluster labels as features
        undersample: if > 0, undersample the majority class by the given factor
        only_two_classes: if True, convert labels to binary classification problem
        include_specifications: if True, include vehicle specifications as features
        histogram_normalizer: if True, normalize histogram features using custom normalizer
        forward_fill: if True, apply forward fill to fill missing values in each vehicle's readouts
        pca: if True, apply PCA to reduce the number of features

        returns: training, validation and test dataloaders
        '''
        train_dataset, val_dataset, test_dataset = ScaniaDataset.get_datasets(data_dir, test_pct, piece_wise_rul, window_size,validation_rate=validation_rate,
                                                                              stored_subsets=stored_subsets,cluster_specifications=cluster_specifications,undersample=undersample,
                                                                              only_two_classes=only_two_classes,include_specifications=include_specifications,
                                                                              histogram_normalizer=histogram_normalizer,forward_fill=forward_fill,pca=pca)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=10,persistent_workers=True,drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=10,persistent_workers=True,drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
        
        return train_dataloader, val_dataloader, test_dataloader  

    @staticmethod
    def get_normalization_parameters(train_subsets):
        '''
        Compute the mean and standard deviation of the training data to normalize the data
        train_subsets: list of subsets to compute the normalization parameters
        '''
        data = pd.concat(train_subsets)
        data_mean = data.drop(columns=['rul']).mean()
        data_std = data.drop(columns=['rul']).std()

        return data_mean, data_std


    def __len__(self):
        return len(self.__sequence_start_indexes)

    def __getitem__(self, idx):
        # get start and end indexes of the sequence
        sequence_start = self.__sequence_start_indexes[idx]
        sequence_end = self.__sequence_end_indexes[idx] 
        length = (sequence_end - sequence_start)
        # if the sequence is shorter than the window size, pad with zeros
        if sequence_end - sequence_start < self.__window_size:
            sample = np.concatenate((np.zeros((self.__window_size-length,self.data.shape[-1])),self.data[sequence_start:sequence_end]))
        else:
            sample = self.data[sequence_start:sequence_end]
        
        sample = torch.FloatTensor(sample).nan_to_num(nan=0.0)
        label = torch.FloatTensor(np.array([self.labels[sequence_end]]))
        # label = torch.FloatTensor(np.array([self.labels[idx]]))

        if self.transform:
            sample = self.transform(sample)

        return sample, label
    
    def get_label_counts(self):
        if self.labels.ndim == 1:
            return np.unique(self.labels,return_counts=True)[1]
        return self.labels[self.__sequence_end_indexes].sum(axis=0)
    
    def get_features(self):
        return self.data.shape[-1]
    
    def get_column_names(self):
        return self.columns