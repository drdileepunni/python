import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler as scaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import copy

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

    def __init__(self, df):
        self.df = df

    def testPerformance(self, data_type, models, ap_tr_list):
        df = self.df

        from sklearn.metrics import confusion_matrix, classification_report
        import matplotlib.pyplot as plt

        print('Arranging Models...')
        # List of models

        # Dictionary of pipelines
        model_dict = {0: 'Stochastic Gradient Boost', 1: 'Gradient Boost', 2: 'RF with batch aggregation', 
                    3: 'Random Forest', 4: 'DT with Batch Aggregation', 5: 'K Nearest Neighbours', 6: 'DecisionTree', 
                    7: 'Logistic Regression', 8: 'Neural Network'}

        roc_ls = []; 
        count_c = 0; count_r = 0
        fig, axs = plt.subplots(3, 3)
        hl_list = []
        
        print('Evaluating performance...')
        case = []
        for idx, model in enumerate(models):        
            X_train, y_train = DatasetProcess(df, model_dict[idx], data_type).dataprocess()
            t = []
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
            confidence_lower, confidence_upper = Boot(y_train, y_train_pred).bootstrap()
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

    def bootstrap(self):
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
            score = roc_auc_score(y_train.values[indices], y_train_pred[indices])
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
