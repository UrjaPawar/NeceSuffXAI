import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from evaluation import Evaluation, measure_kendall_correlation
import warnings
from tqdm import tqdm
import math
warnings.filterwarnings("ignore")





datasets = ["heart_db"]
expl_context = "medical"
nbr_json = "prob"
input_data = {"data": datasets[0], "classifier": "SVM", "fold": "fold1", "explanandum_context": expl_context,
                  "nbr_json": nbr_json}
eval = Evaluation(input_data)

# fi_scores = eval.xai.get_shap_vals(eval.testdf[eval.features])
# try with different neighborhoods, shap, lime

# def shap_approximation(sample, model, A, R):
#
# def lime_approximation():
#
# def cf_approximation():

def r_x_shap(ind, size=100):
    sample = eval.testdf[eval.features].iloc[ind]
    num_samples = len(eval.traindf[eval.features])
    num_features = len(sample)
    k_values = np.random.randint(1, num_features, size=num_samples)
    perturbed_samples = np.array([sample for _ in range(size)])
    probs = []
    for i in range(size):
        indices_to_replace = np.random.choice(num_features, size=k_values[i], replace=False)
        perturbed_samples[i, indices_to_replace] = eval.traindf[eval.features].iloc[i, indices_to_replace]
        num_common_features = num_features-k_values[i]
        probability = (math.comb(num_features - 1, num_common_features) * math.comb(num_features, num_common_features + 1)) / (2 ** num_features)
        probs.append(probability)
    probs = np.array(probs)
    fis = []
    for j,feat in enumerate(eval.features):
        filter_inds = perturbed_samples[:,j]!=sample[feat]
        diffs = eval.clf.predict_proba(perturbed_samples[filter_inds])[:,1]
        diffs = diffs - eval.clf.predict_proba([sample])[0][1]
        fi_j = sum(diffs * probs[filter_inds])
        fis.append(fi_j)
    return fis

def r_x_lime(ind, size=100):
    sample = eval.testdf[eval.features].iloc[ind]
    num_samples = len(eval.traindf[eval.features])
    num_features = len(sample)
    k_values = np.random.randint(1, num_features, size=num_samples)
    perturbed_samples = np.array([sample for _ in range(size)])
    probs = []
    for i in range(size):
        indices_to_replace = np.random.choice(num_features, size=k_values[i], replace=False)
        perturbed_samples[i, indices_to_replace] = eval.traindf[eval.features].iloc[i, indices_to_replace]

    distances = [np.linalg.norm(perturbed_samples[i] - sample, axis=0) for i in range(size)]
    distances = np.array(distances)# Calculate Euclidean distances
    inds_0 = np.where(distances!=0)
    perturbed_samples = perturbed_samples[inds_0]

    weights = 1 / distances[inds_0]

    # Normalize weights to sum up to 1
    weights /= np.sum(weights)
    fis = []
    for j,feat in enumerate(eval.features):
        d = perturbed_samples[:,j]-sample[feat]
        filter_inds = perturbed_samples[:,j]!=sample[feat]
        diffs = eval.clf.predict_proba(perturbed_samples[filter_inds])[:,1]
        diffs = diffs - eval.clf.predict_proba([sample])[0][1]
        fi_j = sum((diffs/d[filter_inds]) * weights[filter_inds])

        fis.append(fi_j)
    return fis
corrs1 = []
corrs2 = []
corrs3 = []
corrs4 = []
corrs5 = []
corrs6 = []
for k in range(len(eval.testdf.iloc[:50])):
    fi_scores_ = r_x_shap(k)
    fi_scores = r_x_lime(k)
    shap_vals = eval.xai.get_shap_vals(eval.testdf[eval.features].iloc[k])
    lime_vals = eval.xai.get_lime(eval.testdf[eval.features].iloc[k],original=True)
    a = [abs(score) for score in lime_vals]
    b = [abs(score) for score in fi_scores]
    a_ = [abs(score) for score in shap_vals]
    b_ = [abs(score) for score in fi_scores_]
    corrs1.append(measure_kendall_correlation(a, b))
    corrs2.append(measure_kendall_correlation(a, a_))
    corrs5.append(measure_kendall_correlation(a, b_))
    corrs6.append(measure_kendall_correlation(b, a_))
    corrs3.append(measure_kendall_correlation(b_, b))
    corrs4.append(measure_kendall_correlation(a_, b_))

print(np.mean(corrs1))
print(np.mean(corrs2))
print(np.mean(corrs3))
print(np.mean(corrs4))
print(np.mean(corrs5))
print(np.mean(corrs6))
# def r_x_lime():
#
# def r_x_cf():
#
# def r_x_nece():
#
# def r_x_suff():

