# %%
import numpy as np
from scipy.special import softmax, expit
from sklearn.metrics import log_loss
from sklearn.dummy import DummyClassifier

# deviance function
def explained_deviance(y_true, y_pred_logits=None, y_pred_probas=None, 
                       returnloglikes=False):
    """Computes explained_deviance score to be comparable to explained_variance"""
    
    assert y_pred_logits is not None or y_pred_probas is not None, "Either the predicted probabilities \
(y_pred_probas) or the predicted logit values (y_pred_logits) should be provided. But neither of the two were provided."
    
    if y_pred_logits is not None and y_pred_probas is None:
        # check if binary or multiclass classification
        if y_pred_logits.ndim == 1: 
            y_pred_probas = expit(y_pred_logits)
        elif y_pred_logits.ndim == 2: 
            y_pred_probas = softmax(y_pred_logits)
        else: # invalid
            raise ValueError(f"logits passed seem to have incorrect shape of {y_pred_logits.shape}")
            
    if y_pred_probas.ndim == 1: y_pred_probas = np.stack([1-y_pred_probas, y_pred_probas], axis=-1)
    
    # compute a null model's predicted probability
    X_dummy = np.zeros(len(y_true))
    y_null_probas = DummyClassifier(strategy='prior').fit(X_dummy,y_true).predict_proba(X_dummy)
    #strategy : {"most_frequent", "prior", "stratified", "uniform",  "constant"}
    # suggestion from https://stackoverflow.com/a/53215317
    llf = -log_loss(y_true, y_pred_probas, normalize=False)
    llnull = -log_loss(y_true, y_null_probas, normalize=False)
    ### McFadden’s pseudo-R-squared: 1 - (llf / llnull)
    explained_deviance = 1 - (llf / llnull)
    ## Cox & Snell’s pseudo-R-squared: 1 - exp((llnull - llf)*(2/nobs))
    # explained_deviance = 1 - np.exp((llnull - llf) * (2 / len(y_pred_probas))) ## TODO, not implemented
    if returnloglikes:
        return explained_deviance, {'loglike_model':llf, 'loglike_null':llnull}
    else:
        return explained_deviance



