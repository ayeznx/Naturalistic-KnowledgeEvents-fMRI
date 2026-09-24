
##### Imports

# General package imports
import os 
import numpy as np

# Encoding model imports
from tikreg import spatial_priors, temporal_priors, models
from sklearn import model_selection


# Prediction performance imports
from scipy.stats import pearsonr

#### Test / train split helper functions

def get_continuous_split(data, start_index, end_index):
    """
        Generates a test / train split using continuous segments of the movie. 
        Takes in data of the shape [TRS x samples]
        Data should be divided using 0:30, 30:60, 60:90, 90:120, or 120: as test set.
        Returns train, test sets
    """
    # Set testing set to match the indexes specified
    test = data[start_index:end_index]
    
    # Concatenate remaining data into training data
    train = np.concatenate((data[:start_index], data[end_index:]), axis=0)
    
    return train, test

##### Banded ridge helper functions 

def generate_priors(features):
    """
        Given a list of feature sets, generates appropriate spatial and temporal priors to be used in tikreg. 
        Returns [feature priors] list, temporal prior
    """
    # A temporal prior is unnecessary for this analysis, so set it to 0 to be ignored in the tikreg function
    temporal_prior = temporal_priors.SphericalPrior(delays=[0])
    
    # Set range of lambdas to be considered when choosing optimal hyperparameters
    ridges = np.logspace(-1,3,5)
    
    # For each feature, generate Spherical Prior object, then append to list
    feature_priors = []
    for feature in features:
        feature_prior = spatial_priors.SphericalPrior(feature, hyparams=ridges)
        feature_priors.append(feature_prior)
    
    return feature_priors, temporal_prior


def isolate_feature(weights, start_index, end_index):
    """
    	Isolates weights for a feature (can be single or multi-dimensional)
    	Assumes input takes the form num_weights x voxel_count
    	Returns array of same shape as input with weights for all but feature of interest zeroed out
    """
    
    isolated_weights = np.zeros(np.shape(weights))
    isolated_weights[start_index:end_index] = weights[start_index:end_index]
    return isolated_weights


# def get_weights_from_kernel(banded_ridge_results, features_train):
#     """
#         Generates weights for use in prediction from the tikreg package's kernel weights
#         Returns array of primal weights
#     """
#     voxelwise_optimal_hyperparameters = banded_ridge_results.get('optima')
#     K = len(features_train)
#     kernel_weights = banded_ridge_results.get('weights')
#     primal_weights = []
#     for voxid, params in enumerate(voxelwise_optimal_hyperparameters):
#         temporal_lambda = params[0]
#         lambdas = params[1:1+K]
#         alpha = params[1+K]

#         X_scaled = np.hstack([(features_train[k] / (lambdas[k]**2)) for k in range(K)])  # 大复制
#         ws = X_scaled.T @ (kernel_weights[:, voxid] * alpha)
#         primal_weights.append(ws)

#     primal_weights = np.asarray(primal_weights).T
#     return primal_weights

def get_weights_from_kernel(banded_ridge_results, features_train):
    """
    Optimized generation of primal weights from kernel weights.
    Groups voxels by optimal hyperparameters to avoid redundant memory allocation 
    and maximize matrix multiplication efficiency.
    """
    # 获取超参数矩阵 (n_voxels, n_params)
    voxelwise_optimal_hyperparameters = banded_ridge_results.get('optima')
    kernel_weights = banded_ridge_results.get('weights') # shape: (n_samples, n_voxels)
    
    n_voxels = kernel_weights.shape[1]
    K = len(features_train)
    
    # 初始化输出权重矩阵
    # 计算总特征维度
    total_feature_dim = sum([f.shape[1] for f in features_train])
    primal_weights = np.zeros((total_feature_dim, n_voxels))
    
    # 1. 找到所有唯一的超参数组合
    # axis=0 表示按行查找唯一值，return_inverse 用于稍后重构索引
    unique_params, inverse_indices = np.unique(voxelwise_optimal_hyperparameters, axis=0, return_inverse=True)
    
    print(f"Optimizing: Found {len(unique_params)} unique hyperparameter sets across {n_voxels} voxels.")

    # 2. 遍历每一组唯一的超参数
    for i, params in enumerate(unique_params):
        # 找到属于这组超参数的所有体素索引
        voxel_mask = (inverse_indices == i)
        
        # 解析参数
        # params结构通常是: [temporal_lambda, feature_lambda_1, ..., feature_lambda_K, alpha]
        # 注意：这里需要根据 tikreg 具体返回的 params 结构微调，假设和你原代码逻辑一致
        lambdas = params[1:1+K] 
        alpha = params[1+K]
        
        # 3. 构建 X_scaled (只做一次!)
        # 利用列表推导式处理除法，然后一次性 hstack
        X_scaled_parts = []
        for k in range(K):
            # 避免除以0，虽然在岭回归中lambda通常不为0
            denom = lambdas[k]**2 if lambdas[k] != 0 else 1e-10
            X_scaled_parts.append(features_train[k] / denom)
        
        X_scaled = np.hstack(X_scaled_parts) # Shape: (n_samples, n_features)
        
        # 4. 批量计算权重
        # kernel_weights[:, voxel_mask] shape: (n_samples, n_subset_voxels)
        # 我们需要对选中的体素应用 alpha
        # broadcasting: (n_samples, n_subset_voxels) * scalar alpha
        current_kernel_weights = kernel_weights[:, voxel_mask] * alpha
        
        # 矩阵乘法: (n_features, n_samples) @ (n_samples, n_subset_voxels) -> (n_features, n_subset_voxels)
        ws_subset = X_scaled.T @ current_kernel_weights
        
        # 5. 填入结果矩阵
        primal_weights[:, voxel_mask] = ws_subset

    return primal_weights


def get_banded_weights(features, features_train, features_test, bold_train, bold_test):
    """
        Takes in social and perceptual features as a list, and test/train splits for features and bold data.
        Feature test and train sets should also be lists.
        Returns weights from fitting a banded ridge regression model
    """

    # Generate appropriate prior objects for tikreg function
    feature_priors, temporal_prior = generate_priors(features)

    # Fit model
    banded_ridge_results = models.estimate_stem_wmvnp(features_train, bold_train, 
                            features_test,bold_test,
                            feature_priors=feature_priors,
                            temporal_prior=temporal_prior,
                            ridges=[1.0], 
                            folds=(1,5),
                            weights=True, # Return beta weights 
                            verbosity=0) # Do not print interim output of tikreg  
    
    # Get correctly formatted weights for prediction
    weights = get_weights_from_kernel(banded_ridge_results, features_train)
    return weights



##### Prediction performance functions


def get_predictions_from_weights(features, subject_weights):
	"""
		Takes in combined feature labeling for held out timepoints and subject weights.
		Returns predictions generated from those weights.
	"""
	return np.matmul(features, subject_weights)

def get_prediction_scores(predicted, test):
    """
            Computes correlation between model prediction and true value for each voxel
            Returns array of Pearson's correlation (r)
    """
    correlations = []

    for voxel in range(predicted.shape[1]):
        predicted_voxel = predicted[:,voxel]
        actual_voxel = test[:,voxel]
        corr, p = pearsonr(predicted_voxel, actual_voxel)
        correlations.append(corr)
        
    return correlations


###### Feature-specific prediction functions

def get_feature_predictions(bold_data, weights, features_test, feature_index):
    """
    """
    feature_predictions = []
    for subject in range(len(bold_data)):
        # Reorient subject data so axes align with feature axes
        subject_data = np.swapaxes(bold_data[subject], 0, 1)
    
        # Generate test / train split for BOLD data
        subject_train, subject_test = model_selection.train_test_split(subject_data, test_size=0.2, random_state=4)
    
        # Get feature specific predictions 
        subject_predictions = get_subject_feature_predictions(feature_index=feature_index, 
                                                      features_test=features_test, 
                                                      subject_test=subject_test, 
                                                      subject_weights=weights[subject])
        feature_predictions.append(subject_predictions)
    return feature_predictions
    

def get_subject_feature_predictions(feature_index, features_test, subject_test, subject_weights):
    """
    Takes in model weights, BOLD data, and feature labels for held out data, then uses weights from a single
    feature to generate and score predictions.
    Returns subject-specific prediction scores using the single feature's weights.
    
    """
    feature_only_weights = isolate_feature(subject_weights, start_index=feature_index, end_index=(feature_index+1))
    predictions = get_predictions_from_weights(features_test, feature_only_weights)
    return get_prediction_scores(predictions, subject_test)



##### Functions for calculating unique variance

def get_unique_variance(base, lesioned):
    """
        Calculate unique variance explained by a model feature.
        Uses the formula u = r_base^2 - r_lesioned^2 
        returns the average unique variance for each voxel.
    """
    base2 = np.square(base)
    lesioned2 = np.square(lesioned)
    u = np.subtract(base2, lesioned2)
    
    return np.mean(u, axis=0)

def get_nonbanded_predictions(features, feature_train, feature_test, bold_train, bold_test):
    """
       	Takes in a single feature space, with test/train splits for features and subject data
        Returns predictions using tikreg for a single feature space
    """

    # Generate appropriate prior objects for tikreg function
    feature_prior, temporal_prior = helpers.generate_priors(features)

    
    # Fit model
    subject_model = models.estimate_stem_wmvnp([feature_train], bold_train, 
                                    [feature_test],bold_test,
                                    feature_priors=[feature_prior],
                                    temporal_prior=temporal_prior,
                                    ridges=[1.0],
                                    folds=(1,5),
                                    predictions=True, # Get predictions for sanity checking
                                    verbosity=0) # Do not print interim output of tikreg

    return subject_model.get('predictions')

