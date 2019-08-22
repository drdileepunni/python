import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler as scaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import copy
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

class changeTypes:
    
    '''
    
    This function drops unwanted colums + change dtypes of the columns we specify it to, 
    into numeric and categoricals
    
    '''
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
    
    def dropAndChange(self, to_drop, numerics, categoricals):
        x = lambda a: pd.to_numeric(a, errors='coerce')
        y = lambda a: a.astype('category')
        changed_df = self.dataFrame.drop(to_drop, axis=1)
        for i in numerics:
            changed_df[i] = x(changed_df[i])
        for i in categoricals:
            changed_df[i] = y(changed_df[i])
        return changed_df 

class changeBad:
    
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        
    """
    
    This function changes all the bad values in the dataframe to normals as described below
        Temp - remove all <95, >104 
        SpO2 - <1 multiply by 100 
        MAP - 0 impute MAP >250 remove 
        RR - >60 remove 
        FiO2 - >1 divide by 100 
        PaO2 - <40, >400 
        PaCO2 - <15, >140 
        drop bicarb column 
        TLC <50, multiply by 1000 
        Platelets <10, multiply by 100000 
        PaO2 lower value 
        A-a remove column 
        Urine output 43 zero values
    
    """
    
    def change(self):
        self.Temperature[self[self.Temperature<95].index] = 98.4
        self.Temperature[self[self.Temperature>104].index] = 98.4
        self.SpO2[self[self.SpO2<=1].index] = (self.SpO2[self[self.SpO2<=1].index])*100
        self.SpO2[self[self.SpO2<40].index] = 98
        self.MAP[self[self.MAP<40].index] = 92
        self.MAP[self[self.MAP>300].index] = 92
        self.RR[self[self.RR>60].index] = 16
        self.FiO2[self[self.FiO2>1].index] = (self.FiO2[self[self.FiO2>1].index])/100
        self.FiO2[self[self.FiO2<0.21].index] = 0.21
        self.PaO2[self[self.PaO2>400].index] = 100
        self.PaO2[self[self.PaO2<40].index] = 100
        self.PaCO2[self[self.PaCO2>140].index] = 38
        self.PaCO2[self[self.PaCO2<15].index] = 38
        self.TLC[self[self.TLC<=50].index] = (self.TLC[self[self.TLC<=50].index])*1000
        self.TLC[self[self.TLC>200000].index] = 9000
        self.Platelets[self[self.Platelets<=10].index] = (self.Platelets[self[self.Platelets<=10].index])*100000
        self.Platelets[self[self.Platelets>2000000].index] = 220000
        self['Urine output'][self[self['Urine output']==0].index] = 1000
        self.GCS[self[self.GCS==2].index] = 3
        return self

class Impute:
    
    '''
    
    Imputes Na values with normal values
    
    '''
    
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        
    def impute(self, num_cols, categoricals):
        fillna_freq = lambda x:x.fillna(x.value_counts().index[0])
        
        for num in num_cols:
            self.dataFrame[num] = self.dataFrame[num].fillna(self.dataFrame[num].median())
            
        for i in categoricals:
            self.dataFrame[i] = fillna_freq(self.dataFrame[i])
            
        return self.dataFrame    

class OHE:
    '''
    One hot encodes the input text
    '''
    
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        
    def encode(self, targ_string):
        fin = pd.get_dummies(self.dataFrame.drop([targ_string], axis = 1), drop_first=True)
        targ = pd.get_dummies(self.dataFrame[targ_string].values, drop_first=True)
        return (fin, targ)

class NNProcess:

    def __init__(self, dataFrame):
        self.dataFrame = dataFrame

    def NNdataprocess(self):
        df_train_unscaled = self.dataFrame
        # Scaling of values
        numerics = df_train_unscaled.iloc[:,list(np.append(0, np.arange(3,24)))].columns
        df_train_scaled = copy.deepcopy(df_train_unscaled)
        for col in numerics:
            df_train_scaled[col] = pd.DataFrame(scaler().fit_transform(pd.DataFrame(df_train_scaled[col])))

        # One hot encoding
        fin, targ = OHE(df_train_scaled).encode('Survival')

        return (fin, targ)

class evaluate:

    def __init__(self, pred, targ):
        self.pred = pred
        self.targ = targ

    def testPerformance(self, data_type, models, ap_tr_list):
        pred = self.pred
        targ = self.targ

        from sklearn.metrics import confusion_matrix, classification_report
        import matplotlib.pyplot as plt

        print('Arranging Models...')
        # List of models

        # Dictionary of pipelines
        model_dict = {0: 'Stochastic Gradient Boost', 1: 'Gradient Boost', 2: 'Random Forest', 
                        3: 'Logistic Regression', 4: 'XGBoost', 5: 'Neural Network'}

        roc_ls = []; 
        count_c = 0; count_r = 0
        fig, axs = plt.subplots(2, 3)
        hl_list = []
        
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
            confidence_lower, confidence_upper = Boot(y_train, y_train_pred).bootstrap('roc')
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

class Boot:

    def __init__(self, y_train, y_train_pred):
        self.y_train = y_train
        self.y_train_pred = y_train_pred

    def bootstrap(self, name):
        y_train = self.y_train
        y_train_pred = self.y_train_pred

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


class DatasetProcess:

    def __init__(self, df, model_name, data_type):
        self.df = df
        self.model_name = model_name
        self.data_type = data_type

    def dataprocess(self):
        df = self.df
        model_name = self.model_name
        data_type = self.data_type
    
        if (model_name!='Neural Network'):
            df_X = df.drop('Survival', axis=1)
            df_y = df['Survival']
            X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, 
                                                                stratify=df_y, random_state=62)
            if (data_type=='train'):
                X_train = X_train
                y_train = y_train
            elif (data_type=='test'):
                X_train = X_test
                y_train = y_test
            elif (data_type=='holdout'):
                X_train = df_X
                y_train = df_y
        else:
            # Scaling of values
            numerics = df.iloc[:,list(np.append(0, np.arange(3,24)))].columns
            df_train_scaled = copy.deepcopy(df)
            for col in numerics:
                df_train_scaled[col] = pd.DataFrame(scaler().fit_transform(pd.DataFrame(df_train_scaled[col])))
            # One hot encoding
            fin_nn, targ_nn = OHE(df_train_scaled).encode('Survival')
            X_train, X_test, y_train, y_test = train_test_split(fin_nn, targ_nn, test_size=0.3, 
                                                            stratify=targ_nn, random_state=62)
            if (data_type=='train'):
                X_train = X_train
                y_train = y_train
            elif (data_type=='test'):
                X_train = X_test
                y_train = y_test
            elif (data_type=='holdout'):
                X_train = fin_nn
                y_train = targ_nn
                
        return (X_train, y_train)

class getFinal:

    def __init__(self, df):
        self.df = df

    def process(self):
        
        df = self.df

        # Defining and naming columns
        df.columns = ['Unnamed: 0', 'Name', 'CPMRN', 'Month of Admission', 'Age', 'Gender',
            'Hospital', 'Surgery', 'Vent mode', 'GCS', 'Temperature', 'HR', 'SpO2',
            'SBP', 'MAP', 'RR', 'FiO2', 'PaO2', 'PaCO2', 'pH', 'A-a gradient',
            'HCO3', 'Hb', 'TLC', 'Platelets', 'K', 'Na', 'Serum Cr', 'Blood Urea',
            'Bili', 'Urine output', 'Lactate', 'INR', 'Survival']
        to_drop = ['Unnamed: 0', 'Name', 'CPMRN', 'SBP', 'A-a gradient', 'Month of Admission', 
                'HCO3', 'Hospital', 'Vent mode']
        numerics = ['Age', 'Temperature', 'GCS', 'HR', 'SpO2', 'Hb', 'TLC', 'Platelets', 'K',
                    'MAP', 'RR', 'FiO2', 'PaO2', 'PaCO2', 'pH', 'Na', 'Serum Cr', 'Blood Urea', 
                    'Bili', 'Urine output', 'Lactate', 'INR']
        categoricals = ['Gender', 'Surgery', 'Survival']

        # Changing types and dropping columns
        df1 = changeTypes(df).dropAndChange(to_drop, numerics, categoricals)

        # Removing and replacing bad values
        df2 = changeBad.change(df1)

        # Imputation of Na values
        df3 = Impute(df2).impute(numerics, categoricals)
        df3_unscaled = pd.DataFrame.copy(df3)

        # Encoding holdout3_unscaled
        survival = {'Alive': 0,'Expired': 1} 
        df3_unscaled['Survival'] = [survival[item] for item in df3_unscaled['Survival']] 

        # Scaling encoding and returning 
        X_h = df3_unscaled.drop('Survival', axis=1)
        y_h = df3_unscaled['Survival']
        X_h_encoded = pd.get_dummies(X_h, drop_first=True)
        cols = X_h_encoded.columns[np.arange(0,22)]
        for col in cols:
            X_h_encoded[col] = pd.DataFrame(scaler().fit_transform(pd.DataFrame(X_h_encoded[col])))
        
        return (X_h_encoded, y_h)

class reliability:

    def __init__(self, y_true, y_score, bins=6, normalize=False):
        self.y_true = y_true
        self.y_score = y_score
        self.bins = bins
        self.normalize = normalize

    def reliability_curve(self):
        y_true = self.y_true
        y_score = self.y_score
        bins = self.bins
        normalize = self.normalize

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

class Calibration_curves:

    def __init__(self, pred, targ, data_type, models, ap_tr_list):
        self.pred = pred
        self.targ = targ
        self.data_type = data_type
        self.models = models
        self.ap_tr_list = ap_tr_list
    
    def getCalibCurves(self):
        pred = self.pred
        targ = self.targ
        data_type = self.data_type
        models = self.models
        ap_tr_list = self.ap_tr_list

        trop = ap_tr_list[0]; ap = ap_tr_list[1]
        import matplotlib.pyplot as plt
        
        # Dictionary of pipelines
        model_dict = {0: 'Stochastic Gradient Boost', 1: 'Gradient Boost',  2: 'Random Forest', 
                    3: 'Logistic Regression', 4: 'XGBoost', 5: 'Neural Network'}
        calib_ls = {}
        count_c = 0; count_r = 0
        fig, axs = plt.subplots(2, 3)
        print('Evaluating performance...')
        case = []
        reliability_scores = {}

        X_train, X_test, y_train, y_test = train_test_split(pred, targ, test_size=0.3, stratify=targ, random_state=62)
        n_bins = 8
        if (data_type == 'test'):
            print('test subset...')
            X_train=X_test
            y_train=y_test
            n_bins = 8
        elif (data_type == 'holdout'):
            print('holdout subset...')
            X_train=pred
            y_train=targ
            n_bins = 5

        flatten = lambda l: [item for sublist in l for item in sublist]
        for idx, model in enumerate(models):        
            # Preparing Dataset
            print('Preparing dataset...')
            # Evaluating
            print('Evaluating...')
            print('\nEstimator: %s' % model_dict[idx])
            # APACHE values
            ap_score_bin_mean, ap_empirical_prob_pos = reliability(y_train, ap, normalize=True, bins=n_bins).reliability_curve()
            ap_scores_not_nan = np.logical_not(np.isnan(ap_empirical_prob_pos))
            # Trop values
            trop_score_bin_mean, trop_empirical_prob_pos = reliability(y_train, trop, normalize=True, bins=n_bins).reliability_curve()
            trop_scores_not_nan = np.logical_not(np.isnan(trop_empirical_prob_pos))
            
            # Predict
            if (model_dict[idx] == 'Logistic Regression'):
                y_score_bin_mean, empirical_prob_pos = reliability(y_train, model.predict_proba(X_train)[:, 1], normalize=True, bins=n_bins).reliability_curve()        
            elif (model_dict[idx] == 'Neural Network'):
                y_score_bin_mean, empirical_prob_pos = reliability(y_train, np.array(flatten(model.predict(X_train))), normalize=True, bins=n_bins).reliability_curve()
            else:
                y_score_bin_mean, empirical_prob_pos = reliability(y_train, model.predict(X_train), normalize=True, bins=n_bins).reliability_curve()
            
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