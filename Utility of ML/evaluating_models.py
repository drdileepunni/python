import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

def testPerformance(pred, targ, data_type, models, ap_tr_list):

    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt

    print('Arranging Models...')
    # List of models

    # Dictionary of pipelines
    model_dict = {0: 'Stochastic Gradient Boost', 1: 'Gradient Boost', 2: 'Random Forest', 
                    3: 'Logistic Regression', 4: 'XGBoost', 5: 'Neural Network'}

    count_c = 0; count_r = 0
    fig, axs = plt.subplots(2, 3)
    
    print('Evaluating performance...')
    case = []
    for idx, model in enumerate(models):        
        X_train, X_test, y_train, y_test = train_test_split(pred, targ, test_size=0.3, stratify=targ, random_state=62)
        t = []
        if (data_type == 'test'):
            X_train=X_test
            y_train=y_test
        elif (data_type == 'holdout'):
            X_train=pred
            y_train=targ
        # Preparing Dataset
        print('Preparing dataset...')

        bag = []
        # Evaluating
        print('Evaluating...')
        bag.append({'model':model_dict[idx]})
        print('\nEstimator: %s' % model_dict[idx])
        # Predict
        if (model_dict[idx] == 'Logistic Regression'):
            y_train_pred = model.predict_proba(X_train)[:, 1]
        else:
            y_train_pred = model.predict(X_train)
        # Find ROC Score
        model_score = roc_auc_score(y_train, y_train_pred)
        print('Score in training set: %s' % model_score)
        bag.append({'score':model_score})
        # Finding confidence intervals
        confidence_lower, confidence_upper = bootstrap(y_train, y_train_pred, 'roc')
        print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
        confidence_lower, confidence_upper))
        bag.append({'CI':(confidence_lower, confidence_upper)})
        # Finding optimal cut off
        fpr, tpr, thresholds = roc_curve(y_train, y_train_pred)
        cutoff = thresholds[np.argmax(tpr-fpr)]
        print("Optimal threshold in training set is: {}".format(cutoff.round(3)))
        # Creating confusion matrix and calculating sensitivity and specificity
        pred_mort = (y_train_pred>cutoff)
        report = classification_report(y_train, (1*pred_mort), output_dict=True)
        bag.append({'CR':report})
        case.append({data_type:bag})
        sur_all = ap_tr_list[2]; ap_all = ap_tr_list[1]; trop_all = ap_tr_list[0]
        fpr_ap, tpr_ap, _ = roc_curve(sur_all, ap_all)
        fpr_tr, tpr_tr, _ = roc_curve(sur_all, trop_all)
        ap_auc = roc_auc_score(sur_all, ap_all)
        tr_auc = roc_auc_score(sur_all, trop_all)
        
        # Plotting ROCs
        axs[count_r, count_c].plot(fpr,tpr, label='AUC Model: {}'.format(round(model_score, 3)))
        axs[count_r, count_c].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[count_r, count_c].set_title(model_dict[idx], fontsize = 10)
        axs[count_r, count_c].plot(fpr_ap, tpr_ap, label='AUC APACHE II: {}'.format(round(ap_auc, 3)))
        axs[count_r, count_c].plot(fpr_tr, tpr_tr, label='AUC TropICS: {}'.format(round(tr_auc, 3)))
        axs[count_r, count_c].legend(loc='lower right')
        for ax in axs.flat:
            ax.set(xlabel='Sensitivity', ylabel='1-Specificity')
        for ax in axs.flat:
            ax.label_outer()
        
        count_c +=1
        if (count_c>2):
            count_r +=1
            count_c = 0

        plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=1.0)
        fig1 = plt.gcf()
        fig1.set_size_inches(16, 12)
        fig1.savefig('ROC_{}.png'.format(data_type), dpi=600)
    return case

def bootstrap(y_train, y_train_pred, name):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_train_pred) - 1, len(y_train_pred))
        if len(np.unique(y_train.values[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        if (name=='roc'):
            score = roc_auc_score(y_train.values[indices], y_train_pred[indices])
        elif (name=='prc'):
            score = precision_recall_curve(y_train.values[indices], y_train_pred[indices])
        elif (name=='ap'):
            score = average_precision_score(y_train.values[indices], y_train_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return(confidence_lower, confidence_upper)