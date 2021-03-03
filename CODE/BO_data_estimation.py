import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from scipy.special import ndtr
from modAL.utils.selection import multi_argmax
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
import warnings

warnings.filterwarnings('ignore')

def EI(mean, std, max_val, tradeoff):
    z = (mean - max_val - tradeoff) / std
    return (mean - max_val - tradeoff)*ndtr(z) + std*norm.pdf(z)

data_labeled = pd.read_csv('descriptors_labeled_020.csv') 
data_unlabeled = pd.read_csv('descriptors_unlabeled_020.csv')

descriptors_cols = ['density','spg','volume','SiOSi_average','SiOSi_gmean','SiOSi_hmean','SiOSi_max','SiOSi_mean','SiOSi_min','SiOSi_skew','SiOSi_std','SiOSi_var','SiO_average','SiO_gmean','SiO_hmean','SiO_max','SiO_mean','SiO_min','SiO_skew','SiO_std','SiO_var','ASA','AV','NASA','NAV','VolFrac','largest_free_sphere','largest_included_sphere','largest_included_sphere_free','max_dim','min_dim','mode_dim']

# scaling is optional
descriptors_labeled_scaled = data_labeled[descriptors_cols]
descriptors_unlabeled_scaled = data_unlabeled[descriptors_cols]

k_target_name = ['k_dft']
g_target_name = ['g_dft']
k_targets = data_labeled[k_target_name]
g_targets = data_labeled[g_target_name]

# parameters
num_GBRmodel = 50
num_instances = 300
num_query_points = 25
v_tradeoff = 0.01
test_size = 0.2 
n_iteration = 1000 
n_crossval = 3 # cross-validation no.

################################################################## for K
k_X_initial = descriptors_labeled_scaled
k_y_initial = k_targets.values.ravel()
k_binsplits = np.linspace(min(k_y_initial[1]),max(k_y_initial),8)

temp_x_train, k_X_test, temp_y_train, k_y_test = train_test_split(k_X_initial, k_y_initial, test_size = test_size)
k_GBR = np.zeros((len(descriptors_unlabeled_scaled),num_GBRmodel))
k_GBR_test = np.zeros((2,len(k_X_test),num_GBRmodel))
k_GBR_score = np.zeros((4,num_GBRmodel))
k_GBR_trainval = np.zeros((2,len(temp_x_train),num_GBRmodel))

param_grid = {
    'learning_rate': np.arange(0.01, 0.1, 0.01),
    'max_bin': np.arange(20, 200, 20),
    'min_data_in_leaf': np.arange(4, 8),
    'max_depth': np.arange(3, 6),
    'num_leaves': np.arange(25, 200, 25),
    'bagging_freq': np.arange(6, 15),
    'bagging_fraction': np.arange(0.1, 0.9, 0.1),
    'feature_fraction': np.arange(0.1, 0.9, 0.1),
}
  
for iter in range(0,num_GBRmodel,1):

    while True:   
        print(iter)
        k_categorized = np.digitize(k_y_initial, bins=k_binsplits)
        temp_k_X_train, k_X_test, temp_k_y_train, k_y_test = train_test_split(k_X_initial, k_y_initial, stratify=k_categorized, test_size = test_size)
        
        k_binsplits = np.linspace(min(k_y_initial[1]),max(k_y_initial),8)
        train_k_categorized = np.digitize(temp_k_y_train, bins=k_binsplits)
        k_X_train, k_X_val, k_y_train, k_y_val = train_test_split(temp_k_X_train, temp_k_y_train, stratify=train_k_categorized, test_size = 0.2)
        split_index = []
        for iiter in range(0,len(k_y_train),1):
            temp_k_y = temp_k_y_train[iiter]
            if temp_k_y in k_y_train.tolist():
                split_index.append(-1)
            else: 
                split_index.append(0)
        
        pds = PredefinedSplit(test_fold = split_index)
            
        clf_k = lgb.LGBMRegressor()
        k_optimized_model = RandomizedSearchCV(estimator=clf_k, param_distributions=param_grid, n_iter=n_iteration, cv=pds, verbose=3, n_jobs=-1, return_train_score='True')
        k_optimized_model.fit(temp_k_X_train, temp_k_y_train)
    
        temp_mean_train = k_optimized_model.cv_results_['mean_train_score'][k_optimized_model.best_index_]
        temp_std_train = k_optimized_model.cv_results_['std_train_score'][k_optimized_model.best_index_]
        temp_mean_val = k_optimized_model.cv_results_['mean_test_score'][k_optimized_model.best_index_]
        temp_std_val = k_optimized_model.cv_results_['std_test_score'][k_optimized_model.best_index_]
        
    k_GBR_score[:,iter] = [temp_mean_train, temp_std_train, temp_mean_val, temp_std_val]
    k_GBR_test[:,:,iter] = [k_optimized_model.predict(k_X_test), k_y_test]
    k_GBR_trainval[:,:,iter] = [temp_k_y_train, k_optimized_model.predict(temp_k_X_train)]
    k_GBR[:,iter] = k_optimized_model.predict(descriptors_unlabeled_scaled)
    
    ## temporal accuracy checking 
    correlation_temp2 = np.corrcoef(k_optimized_model.predict(k_X_test), k_y_test)
    correlation_xy_temp2 = correlation_temp2[0,1]
    r2_temp2 = correlation_xy_temp2**2

k_GBR_mean = np.mean(k_GBR,axis=1)
k_GBR_std = np.std(k_GBR,axis=1)

######################################## for query
k_ei = EI(k_GBR_mean, k_GBR_std, np.max(k_y_initial), tradeoff=v_tradeoff)
k_query_idx = multi_argmax(k_ei, n_instances=num_instances)
k_query_pcodid = data_unlabeled['name'].iloc[k_query_idx]

################################################################## for G# assembling initial training set
# assembling initial training set
g_X_initial = descriptors_labeled_scaled
g_y_initial = g_targets.values.ravel()
g_binsplits = np.linspace(min(g_y_initial),max(g_y_initial),8)

temp_x_train, g_X_test, temp_y_train, g_y_test = train_test_split(g_X_initial, g_y_initial, test_size = test_size)

g_GBR = np.zeros((len(descriptors_unlabeled_scaled),num_GBRmodel))
g_GBR_test = np.zeros((2,len(g_X_test),num_GBRmodel))
g_GBR_score = np.zeros((4,num_GBRmodel))
g_GBR_trainval = np.zeros((2,len(temp_x_train),num_GBRmodel))

param_grid = {
    'learning_rate': np.arange(0.01, 0.1, 0.01),
    'max_bin': np.arange(20, 200, 10),
    'min_data_in_leaf': np.arange(4, 10),
    'max_depth': np.arange(3, 8),
    'num_leaves': np.arange(25, 200, 25),
    'bagging_freq': np.arange(6, 15),
    'bagging_fraction': np.arange(0.1, 0.9, 0.1),
    'feature_fraction': np.arange(0.1, 0.9, 0.1),
}

for iter in range(0,num_GBRmodel,1):

    while True:
        
        print(iter)
        g_categorized = np.digitize(g_y_initial, bins=g_binsplits)
        temp_g_X_train, g_X_test, temp_g_y_train, g_y_test = train_test_split(g_X_initial, g_y_initial, stratify=g_categorized, test_size = test_size)
    
        g_binsplits = np.linspace(min(g_y_initial),max(g_y_initial),8)
        train_g_categorized = np.digitize(temp_g_y_train, bins=g_binsplits)
        g_X_train, g_X_val, g_y_train, g_y_val = train_test_split(temp_g_X_train, temp_g_y_train, stratify=train_g_categorized, test_size = 0.2)
        split_index = []
        for iiter in range(0,len(g_y_train),1):
            temp_g_y = temp_g_y_train[iiter]
            if temp_g_y in g_y_train.tolist():
                split_index.append(-1)
            else: 
                split_index.append(0)
  
        pds = PredefinedSplit(test_fold = split_index)

        clf_g = lgb.LGBMRegressor()
        g_optimized_model = RandomizedSearchCV(estimator=clf_g, param_distributions=param_grid, n_iter=n_iteration, cv=pds, verbose=3, n_jobs=-1, return_train_score='True')
        g_optimized_model.fit(temp_g_X_train, temp_g_y_train)
        
        temp_mean_train = g_optimized_model.cv_results_['mean_train_score'][g_optimized_model.best_index_]
        temp_std_train = g_optimized_model.cv_results_['std_train_score'][g_optimized_model.best_index_]
        temp_mean_val = g_optimized_model.cv_results_['mean_test_score'][g_optimized_model.best_index_]
        temp_std_val = g_optimized_model.cv_results_['std_test_score'][g_optimized_model.best_index_]

    g_GBR_score[:,iter] = [temp_mean_train, temp_std_train, temp_mean_val, temp_std_val]
    g_GBR_test[:,:,iter] = [g_optimized_model.predict(g_X_test), g_y_test]
    g_GBR_trainval[:,:,iter] = [temp_g_y_train, g_optimized_model.predict(temp_g_X_train)]
    g_GBR[:,iter] = g_optimized_model.predict(descriptors_unlabeled_scaled)
    
    ## temporal accuracy checking 
    correlation_temp2 = np.corrcoef(g_optimized_model.predict(g_X_test), g_y_test)
    correlation_xy_temp2 = correlation_temp2[0,1]
    r2_temp2 = correlation_xy_temp2**2

g_GBR_mean = np.mean(g_GBR,axis=1)
g_GBR_std = np.std(g_GBR,axis=1)

######################################## for query
g_ei = EI(g_GBR_mean, g_GBR_std, np.max(g_y_initial), tradeoff=v_tradeoff)
g_query_idx = multi_argmax(g_ei, n_instances=num_instances)
g_query_pcodid = data_unlabeled['name'].iloc[g_query_idx]
