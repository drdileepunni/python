import numpy as np
from sklearn.model_selection import train_test_split

def reliability_curve(y_true, y_score, bins=10, normalize=False):

    if normalize:  # Normalize scores into bin [0, 1]
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    for i, threshold in enumerate(bin_centers):
        # determine all samples where y_score falls into the i-th bin
        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                                y_score <= threshold + bin_width / 2)
        # Store mean y_score and mean empirical probability of positive class
        y_score_bin_mean[i] = y_score[bin_idx].mean()
        empirical_prob_pos[i] = y_true[bin_idx].mean()
    return y_score_bin_mean, empirical_prob_pos

def getCalibCurves(pred, targ, data_type, models, ap_tr_list):
    n_bins=4
    trop = ap_tr_list[0]; ap = ap_tr_list[1]
    import matplotlib.pyplot as plt
    
    # Dictionary of pipelines
    model_dict = {0: 'Stochastic Gradient Boost', 1: 'Gradient Boost',  2: 'Random Forest', 
                3: 'Logistic Regression', 4: 'XGBoost', 5: 'Neural Network'}
    calib_ls = {}
    count_c = 0; count_r = 0
    fig, axs = plt.subplots(2, 3)
    case = []
    reliability_scores = {}
    n_bins = 4
    X_train = pred
    y_train = targ

    flatten = lambda l: [item for sublist in l for item in sublist]
    for idx, model in enumerate(models):        
        # Preparing Dataset
        # Evaluating
        # APACHE values
        ap_score_bin_mean, ap_empirical_prob_pos = reliability_curve(y_train, ap, normalize=True, bins=n_bins)
        ap_scores_not_nan = np.logical_not(np.isnan(ap_empirical_prob_pos))
        # Trop values
        trop_score_bin_mean, trop_empirical_prob_pos = reliability_curve(y_train, trop, normalize=True, bins=n_bins)
        trop_scores_not_nan = np.logical_not(np.isnan(trop_empirical_prob_pos))
        
        # Predict
        if (model_dict[idx] == 'Logistic Regression'):
            y_score_bin_mean, empirical_prob_pos = reliability_curve(y_train, model.predict_proba(X_train)[:, 1], normalize=True, bins=n_bins)      
        elif (model_dict[idx] == 'Neural Network'):
            y_score_bin_mean, empirical_prob_pos = reliability_curve(y_train, np.array(flatten(model.predict(X_train))), normalize=True, bins=n_bins)
        else:
            y_score_bin_mean, empirical_prob_pos = reliability_curve(y_train, model.predict(X_train), normalize=True, bins=n_bins)
        
        # Plotting 
        axs[count_r, count_c].plot([0.0, 1.0], [0.0, 1.0], 'k', label="Perfect", ls='--')
        scores_not_nan = np.logical_not(np.isnan(empirical_prob_pos))
        axs[count_r, count_c].plot(y_score_bin_mean[scores_not_nan],empirical_prob_pos[scores_not_nan], 
                                label=model_dict[idx], lw=2)
        axs[count_r, count_c].plot(ap_score_bin_mean[ap_scores_not_nan],ap_empirical_prob_pos[ap_scores_not_nan], 
                                label='APACHE')
        axs[count_r, count_c].plot(trop_score_bin_mean[trop_scores_not_nan],trop_empirical_prob_pos[trop_scores_not_nan], 
                                label='TropICS')
        axs[count_r, count_c].set_title(model_dict[idx], fontsize = 1)
        axs[count_r, count_c].legend(loc='upper left')
        calib_ls[model_dict[idx]] = (y_score_bin_mean[scores_not_nan],empirical_prob_pos[scores_not_nan])
        
        for ax in axs.flat:
            ax.set(xlabel='Predicted probability', ylabel='Observed probability')
        for ax in axs.flat:
            ax.label_outer()

        count_c +=1
        if (count_c>2):
            count_r +=1
            count_c = 0
        
        fig1 = plt.gcf()
        fig1.set_size_inches(16, 8)
        fig1.savefig('Calib_{}.png'.format(data_type), dpi=600)
    return calib_ls