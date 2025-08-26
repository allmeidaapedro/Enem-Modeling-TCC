'''
This script aims to provide functions that will turn the modelling process easier
'''

'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Modelling.
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, TargetEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc, brier_score_loss, accuracy_score, f1_score, recall_score, precision_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import time

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')

# Definições de cores -> todas estão numa escala de mais escura para mais clara.
VERMELHO_FORTE = '#461220'
CINZA1, CINZA2, CINZA3 = '#231F20', '#414040', '#555655'
CINZA4, CINZA5, CINZA6 = '#646369', '#76787B', '#828282'
CINZA7, CINZA8, CINZA9 = '#929497', '#A6A6A5', '#BFBEBE'
AZUL1, AZUL2, AZUL3, AZUL4 = '#174A7E', '#4A81BF', '#94B2D7', '#94AFC5'
VERMELHO1, VERMELHO2, VERMELHO3, VERMELHO4, VERMELHO5 = '#DB0527', '#E23652', '#ED8293', '#F4B4BE', '#FBE6E9'
VERDE1, VERDE2 = '#0C8040', '#9ABB59'
LARANJA1 = '#F79747'
AMARELO1, AMARELO2, AMARELO3, AMARELO4, AMARELO5 = '#FFC700', '#FFCC19', '#FFEB51', '#FFE37F', '#FFEEB2'
BRANCO = '#FFFFFF'


class ColumnDropper(BaseEstimator, TransformerMixin):
    '''
    A transformer class to drop specified columns from a DataFrame.

    Attributes:
        to_drop (list): A list of column names to be dropped.

    Methods:
        fit(X, y=None): Fit the transformer to the data. This method does nothing and is only provided to comply with the Scikit-learn API.
        transform(X): Transform the input DataFrame by dropping specified columns.
    '''

    def __init__(self, to_drop):
        '''
        Initialize the ColumnDropper transformer.

        Args:
            to_drop (list): A list of column names to be dropped.
        '''
        self.to_drop = to_drop

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        This method does nothing and is only provided to comply with the Scikit-learn API.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by dropping specified columns.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after dropping specified columns.
        '''
        # Certify that only present columns will be dropped.
        self.to_drop = [col for col in self.to_drop if col in X.columns]
        
        # Drop the specified columns.
        return X.drop(columns=self.to_drop)
    

class ClassificationFeatureEngineer(BaseEstimator, TransformerMixin):
    '''
    A transformer class for performing feature engineering on absence-related data.

    Methods:
        fit(X, y=None): Fit the transformer to the data. This method does nothing and is only provided to comply with the Scikit-learn API.
        transform(X): Transform the input DataFrame by engineering absence-related features.
    '''

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        This method does nothing and is only provided to comply with the Scikit-learn API.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by engineering absence-related features.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after engineering absence-related features.
        '''
        X_copy = X.copy()

        return X_copy


class OneHotFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    A transformer class for one-hot encoding specified categorical variables.

    Attributes:
        to_encode (list): A list of column names to be one-hot encoded.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by one-hot encoding specified columns.
    '''

    def __init__(self, to_encode):
        '''
        Initialize the OneHotFeatureEncoder transformer.

        Args:
            to_encode (list): A list of column names to be one-hot encoded.
        '''
        self.to_encode = to_encode
        self.encoder = OneHotEncoder(drop='first',
                                     sparse_output=False,
                                     dtype=np.int8,
                                     handle_unknown='ignore',
                                     feature_name_combiner='concat')

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        self.encoder.fit(X[self.to_encode])
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by one-hot encoding specified columns.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after one-hot encoding specified columns.
        '''
        # One-hot encode the columns.
        X_one_hot = self.encoder.transform(X[self.to_encode])

        # Create a dataframe for the one-hot encoded data.
        one_hot_df = pd.DataFrame(X_one_hot,
                                  columns=self.encoder.get_feature_names_out(self.to_encode))

        # Reset for mapping and concatenate constructing a final dataframe of features.
        X_reset = X.reset_index(drop=True)

        return pd.concat([X_reset.drop(columns=self.to_encode), one_hot_df], axis=1)
    

class StandardFeatureScaler(BaseEstimator, TransformerMixin):
    '''
    A transformer class for standard scaling specified numerical features and retaining feature names.

    Attributes:
        to_scale (list): A list of column names to be scaled.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by standard scaling specified columns and retaining feature names.
    '''
    def __init__(self, to_scale):
        '''
        Initialize the StandardFeatureScaler transformer.

        Args:
            to_scale (list): A list of column names to be scaled.
        '''
        self.to_scale = to_scale
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        self.scaler.fit(X[self.to_scale])
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by standard scaling specified columns and retaining feature names.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after standard scaling specified columns and retaining feature names.
        '''
        # Scale the columns.
        X_scaled = self.scaler.transform(X[self.to_scale])
        
        # Create a dataframe for the scaled data.
        scaled_df = pd.DataFrame(X_scaled,
                                 columns=self.scaler.get_feature_names_out(self.to_scale))
        
        # Reset for mapping and concatenated constructing a final dataframe of features.
        X_reset = X.reset_index(drop=True)
        
        return pd.concat([X_reset.drop(columns=self.to_scale), scaled_df], axis=1)
    
    
    
class OrdinalFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    A transformer class for ordinal encoding specified categorical features and retaining feature names.

    Attributes:
        to_encode (dict): A dictionary where keys are column names and values are lists representing the desired category orders.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by ordinal encoding specified columns and retaining feature names.
    '''
    def __init__(self, to_encode):
        '''
        Initialize the OrdinalFeatureEncoder transformer.

        Args:
            to_encode (dict): A dictionary where keys are column names and values are lists representing the desired category orders.
        '''
        self.to_encode = to_encode
        self.encoder = OrdinalEncoder(dtype=np.int8, 
                                      categories=[to_encode[col] for col in to_encode])

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        self.encoder.fit(X[list(self.to_encode.keys())])
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by ordinal encoding specified columns and retaining feature names.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after ordinal encoding specified columns and retaining feature names.
        '''
        # Ordinal encode the columns.
        X_ordinal = self.encoder.transform(X[list(self.to_encode.keys())])
        
        # Create a dataframe for the ordinal encoded data.
        ordinal_encoded_df = pd.DataFrame(X_ordinal,
                                          columns=self.encoder.get_feature_names_out(list(self.to_encode.keys())))
        
        # Reset for mapping and concatenated constructing a final dataframe of features.
        X_reset = X.reset_index(drop=True)
        
        return pd.concat([X_reset.drop(columns=list(self.to_encode.keys())), ordinal_encoded_df], axis=1)
    

class TargetFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    A transformer class for target encoding specified categorical variables.

    Attributes:
        to_encode (list): A list of column names to be target encoded.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by target encoding specified columns.
    '''

    def __init__(self, to_encode):
        '''
        Initialize the TargetFeatureEncoder transformer.

        Args:
            to_encode (list): A list of column names to be target encoded.
        '''
        self.to_encode = to_encode
        self.encoder = TargetEncoder()

    def fit(self, X, y):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like): Target labels.

        Returns:
            self: Returns an instance of self.
        '''
        self.encoder.fit(X[self.to_encode], y)
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by target encoding specified columns.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after target encoding specified columns.
        '''
        # Target encode the columns.
        X_target = self.encoder.transform(X[self.to_encode])

        # Create a dataframe for the target encoded data.
        target_df = pd.DataFrame(X_target,
                                 columns=self.encoder.get_feature_names_out(self.to_encode))

        # Reset for mapping and concatenate constructing a final dataframe of features.
        X_reset = X.reset_index(drop=True)

        return pd.concat([X_reset.drop(columns=self.to_encode), target_df], axis=1)


# Classification modelling.

def classification_kfold_cv(models, X_train, y_train, n_folds=5):
    '''
    Evaluate multiple machine learning models using k-fold cross-validation.

    This function evaluates a dictionary of machine learning models by training each model on the provided training data
    and evaluating their performance using k-fold cross-validation. The evaluation metric used is ROC-AUC score.

    Args:
        models (dict): A dictionary where the keys are model names and the values are instantiated machine learning model objects.
        X_train (array-like): The training feature data.
        y_train (array-like): The corresponding target labels for the training data.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for each model, including their average validation scores
                  and training scores.

    Raises:
        CustomException: If an error occurs while evaluating the models.

    '''


    try:
        # Stratified KFold in order to maintain the target proportion on each validation fold - dealing with imbalanced target.
        stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Dictionaries with validation and training scores of each model for plotting further.
        models_val_scores = dict()
        models_train_scores = dict()

        for model in models:
            # Get the model object from the key with his name.
            model_instance = models[model]

            # Measure training time.
            start_time = time.time()
            
            # Fit the model to the training data.
            model_instance.fit(X_train, y_train)

            end_time = time.time()
            training_time = end_time - start_time

            # Make predictions on training data and evaluate them.
            y_train_pred = model_instance.predict(X_train)
            train_score = roc_auc_score(y_train, y_train_pred)

            # Evaluate the model using k-fold cross validation, obtaining a robust measurement of its performance on unseen data.
            val_scores = cross_val_score(model_instance, X_train, y_train, scoring='roc_auc', cv=stratified_kfold)
            avg_val_score = val_scores.mean()
            val_score_std = val_scores.std()

            # Add the model scores to the validation and training scores dictionaries.
            models_val_scores[model] = avg_val_score
            models_train_scores[model] = train_score

            # Print the results.
            print(f'{model} results: ')
            print('-'*50)
            print(f'Training score: {train_score}')
            print(f'Average validation score: {avg_val_score}')
            print(f'Standard deviation: {val_score_std}')
            print(f'Training time: {round(training_time, 5)} seconds')
            print()

        # Convert scores to a dataframe
        val_df = pd.DataFrame(list(models_val_scores.items()), columns=['model', 'avg_val_score'])
        train_df = pd.DataFrame(list(models_train_scores.items()), columns=['model', 'train_score'])
        eval_df = val_df.merge(train_df, on='model')

        # Sort the dataframe by the best ROC-AUC score.
        eval_df  = eval_df.sort_values(['avg_val_score'], ascending=False).reset_index(drop=True)
        
        return eval_df
    
    except Exception as e:
        raise(CustomException(e, sys))
    

def plot_classification_kfold_cv(eval_df, figsize=(20, 7), bar_width=0.35, title_size=15,
                             title_pad=30, label_size=11, labelpad=20, legend_x=0.08, legend_y=1.08):
    '''
    Plot classification performance using k-fold cross-validation.

    Parameters:
        eval_df (DataFrame): DataFrame containing evaluation metrics for different models.
        figsize (tuple, optional): Figure size (width, height). Defaults to (20, 7).
        bar_width (float, optional): Width of bars in the plot. Defaults to 0.35.
        title_size (int, optional): Font size of the plot title. Defaults to 15.
        title_pad (int, optional): Padding of the plot title. Defaults to 30.
        label_size (int, optional): Font size of axis labels. Defaults to 11.
        labelpad (int, optional): Padding of axis labels. Defaults to 20.
        legend_x (float, optional): x-coordinate of legend position. Defaults to 0.08.
        legend_y (float, optional): y-coordinate of legend position. Defaults to 1.08.

    Raises:
        CustomException: Raised if an unexpected error occurs.

    Returns:
        None
    '''
    try:
        # Plot each model and their train and validation (average) scores.
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(eval_df['model']))
        y = np.arange(len(eval_df['train_score']))

        val_bars = ax.bar(x - bar_width/2, eval_df['avg_val_score'], bar_width, label='Val score', color=AZUL1)
        train_bars = ax.bar(x + bar_width/2, eval_df['train_score'], bar_width, label='Train score', color=CINZA7)

        ax.set_xlabel('Model', color=CINZA1, labelpad=labelpad, fontsize=label_size, loc='left')
        ax.set_ylabel('ROC-AUC', color=CINZA1, labelpad=labelpad, fontsize=label_size, loc='top')
        ax.set_title("Models' performances", fontweight='bold', fontsize=title_size, pad=title_pad, color=CINZA1, loc='left')
        ax.set_xticks(x, eval_df['model'], rotation=0, color=CINZA1, fontsize=10.8)
        ax.tick_params(axis='y', color=CINZA1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)

        # Add scores on top of each bar
        for bar in val_bars + train_bars:
            height = bar.get_height()
            plt.annotate('{}'.format(round(height, 2)),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', color=CINZA1)

        # Define handles and labels for the legend with adjusted sizes
        handles = [plt.Rectangle((0,0), 0.1, 0.1, fc=AZUL1, edgecolor = 'none'),
                plt.Rectangle((0,0), 0.1, 0.1, fc=CINZA7, edgecolor = 'none')]
        labels = ['Val score', 'Train score']
            
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(legend_x, legend_y), frameon=False, ncol=2, fontsize=10)
    
    except Exception as e:
        raise CustomException(e, sys)
    

def calibrate_model(model, X_val, y_val):
    # Predicting probabilities on the validation set
    y_prob_val = model.predict_proba(X_val)[:, 1]

    # Calculating the calibration curve and Brier score
    fraction_of_positives, mean_predicted_prob = calibration_curve(y_val, y_prob_val, n_bins=10)
    print(f'Brier score: {brier_score_loss(y_val, y_prob_val)}')

    # Plotting the Calibration Plot (before calibration)
    plt.plot(mean_predicted_prob, fraction_of_positives, "s-", label="Model (uncalibrated)")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title('Calibration plot (before calibration)')
    plt.legend()
    plt.show()

    # Calibrating the model using Isotonic Regression
    calibrated_model_isotonic = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated_model_isotonic.fit(X_val, y_val)

    # Predicting probabilities on the validation set after calibration
    y_prob_val_isotonic = calibrated_model_isotonic.predict_proba(X_val)[:, 1]

    # Calculating the calibration curve and Brier score after calibration
    fraction_of_positives_isotonic, mean_predicted_prob_isotonic = calibration_curve(y_val, y_prob_val_isotonic, n_bins=10)
    print(f'Brier score: {brier_score_loss(y_val, y_prob_val_isotonic)}')

    # Plotting the Calibration Plot (after calibration)
    plt.plot(mean_predicted_prob_isotonic, fraction_of_positives_isotonic, "s-", label="Model (with Isotonic Regression)")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title('Calibration plot (after Isotonic Regression)')
    plt.legend()
    plt.show()

    return calibrated_model_isotonic


def evaluate_classifier(y_true, y_pred, probas):
    '''
    Evaluate the performance of a binary classifier and visualize the results.

    This function calculates and displays various evaluation metrics for a binary classifier,
    including the classification report, confusion matrix, ROC curve and AUC, brier score, gini and ks.

    Args:
    - y_true (pd.series): True binary labels.
    - y_pred (pd.series): Predicted binary labels.
    - probas (pd.series): Predicted probabilities of positive class.

    Returns:
    - model_metrics (pd.DataFrame): A dataframe containing the classification metrics for the passed set.

    Raises:
    - CustomException: If an error occurs during evaluation.
    '''

    try:
        # Print classification report and calculate its metrics to include in the final metrics df.
        print(classification_report(y_true, y_pred))
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Calculate and print brier score, gini and ks.
        brier_score = brier_score_loss(y_true, probas)
        print(f'Brier Score: {round(brier_score, 2)}')
        
        fpr, tpr, thresholds = roc_curve(y_true, probas)
        roc_auc = roc_auc_score(y_true, probas)
        gini = 2 * roc_auc - 1
        print(f'Gini: {round(gini, 2)}')
        
        scores = pd.DataFrame()
        scores['actual'] = y_true.reset_index(drop=True)
        scores['absent_probability'] = probas
        sorted_scores = scores.sort_values(by=['absent_probability'], ascending=False)
        sorted_scores['cum_present'] = (1 - sorted_scores['actual']).cumsum() / (1 - sorted_scores['actual']).sum()
        sorted_scores['cum_absent'] = sorted_scores['actual'].cumsum() / sorted_scores['actual'].sum()
        sorted_scores['ks'] = np.abs(sorted_scores['cum_absent'] - sorted_scores['cum_present'])
        ks = sorted_scores['ks'].max()
        
        print(f'KS: {round(ks, 2)}')
        
        # Confusion matrix.
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot = True, fmt = 'd')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Values')
        plt.ylabel('Real Values')
        plt.show()
        
        # Plot ROC Curve and ROC-AUC.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}', color=AZUL1)
        ax.plot([0, 1], [0, 1], linestyle='--', color=CINZA4)  # Random guessing line.
        ax.set_xlabel('False Positive Rate', fontsize=10.8, color=CINZA1, labelpad=20, loc='left')
        ax.set_ylabel('True Positive Rate', fontsize=10.8, color=CINZA1, labelpad=20, loc='top')
        ax.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=CINZA1)
        ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=CINZA1)
        ax.set_title('Receiver operating characteristic (ROC) curve', fontweight='bold', fontsize=12, color=CINZA1, pad=20, loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
        ax.legend()
    
        # PR AUC Curve and score.

        # Calculate model precision-recall curve.
        p, r, _ = precision_recall_curve(y_true, probas)
        pr_auc = auc(r, p)
        
        # Plot the model precision-recall curve.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(r, p, marker='.', label=f'PR AUC = {pr_auc:.2f}', color=AZUL1)
        ax.set_xlabel('Recall', fontsize=10.8, color=CINZA1, labelpad=20, loc='left')
        ax.set_ylabel('Precision', fontsize=10.8, color=CINZA1, labelpad=20, loc='top')
        ax.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=CINZA1)
        ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=CINZA1)
        ax.set_title('Precision-recall (PR) curve', fontweight='bold', fontsize=12, color=CINZA1, pad=20, loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
        ax.legend()

        # Construct a DataFrame with metrics for passed sets.
        model_metrics = pd.DataFrame({
                                    'Metric': ['Accuracy',
                                               'Precision',
                                               'Recall',
                                               'F1-Score',
                                               'ROC-AUC',
                                               'KS',
                                               'Gini',
                                               'PR-AUC',
                                               'Brier'],
                                    'Value': [accuracy, 
                                              precision, 
                                              recall,
                                              f1,
                                              roc_auc,
                                              ks,
                                              gini, 
                                              pr_auc,
                                              brier_score,
                                              ],
                                    })
        
        return model_metrics

    except Exception as e:
        raise CustomException(e, sys)


def plot_probability_distributions(y_true, probas):
    '''
    Plots the kernel density estimate (KDE) of predicted probabilities for absent and present candidates.

    Parameters:
    - y_true (array-like): The true class labels (0 for present, 1 for absent).
    - probas (array-like): Predicted probabilities for the positive class (absent candidates).

    Raises:
    - CustomException: Raised if an unexpected error occurs during plotting.

    Example:
    ```python
    plot_probability_distributions(y_true, probas)
    ```

    Dependencies:
    - pandas
    - seaborn
    - matplotlib

    Note:
    - The function assumes the existence of color constants VERMELHO_FORTE, CINZA7, CINZA1, CINZA9.

    The function creates a KDE plot illustrating the distribution of predicted probabilities for absent and present candidates.
    It provides visual insights into the model's ability to distinguish between the two classes.

    '''
    try:
        probas_df = pd.DataFrame({'Probabilidade de Abstenção': probas,
                                'Abstenção': y_true})

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.kdeplot(data=probas_df, x='Probabilidade de Abstenção', hue='Abstenção', fill=True, ax=ax, palette=[CINZA7, VERMELHO_FORTE])
        ax.set_title('Distribuição das probabilidades preditas - ausentes e presentes', fontweight='bold', fontsize=12, color=CINZA1, pad=20, loc='left')
        ax.set_xlabel('Probabilidades Preditas', fontsize=10.8, color=CINZA1, labelpad=20, loc='left')
        ax.set_ylabel('Densidade', fontsize=10.8, color=CINZA1, labelpad=20, loc='top')
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                    color=CINZA1)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
                    ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6'],
                    color=CINZA1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
        
        handles = [plt.Rectangle((0,0), 0.1, 0.1, fc=VERMELHO_FORTE, edgecolor = 'none'),
                plt.Rectangle((0,0), 0.1, 0.1, fc=CINZA7, edgecolor = 'none')]
        labels = ['Ausentes', 'Presentes']
            
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.17, 1.05), frameon=False, ncol=2, fontsize=10)

        
    except Exception as e:
        raise CustomException(e, sys)


def probability_scores_ordering(y_true, probas):
    '''
    Order and visualize the probability scores in deciles based on predicted probabilities and true labels.

    Parameters:
    - y_true (pd.Series): Actual target values for the set. 1 is absent and 0 is present.
    - probas (pd.Series): Predicted probabilities of being absent for the passed set.

    Returns:
    - None: Plots the probability scores ordering.

    Raises:
    - CustomException: An exception is raised if an error occurs during the execution.
    
    Example:
    ```python
    probability_scores_ordering(y_test, probas)
    ```
    '''
    try:
        # Add some noise to the predicted probabilities and round them to avoid duplicate problems in bin limits.
        noise = np.random.uniform(0, 0.0001, size=probas.shape)
        probas += noise
        #probas = round(probas, 10)
        
        # Create a DataFrame with the predicted probabilities of being absent and actual values.
        probas_actual_df = pd.DataFrame({'probabilities': probas, 'actual': y_true.reset_index(drop=True)})
        
        # Sort the probas_actual_df by probabilities.
        probas_actual_df = probas_actual_df.sort_values(by='probabilities', ascending=True)
        
        # Calculate the deciles.
        probas_actual_df['deciles'] = pd.qcut(probas_actual_df['probabilities'], q=10, labels=False, duplicates='drop')
        
        # Calculate the absent rate per decile.
        decile_df = probas_actual_df.groupby(['deciles'])['actual'].mean().reset_index().rename(columns={'actual': 'absent_rate'})
        
        # Plot probability scores ordering.
        # Plot bar graph of deciles vs event rate.
        fig, ax = plt.subplots(figsize=(12, 3))
        
        bars = ax.bar(decile_df['deciles'], decile_df['absent_rate'], color=VERMELHO_FORTE)
        
        ax.set_title('Ordenação dos scores de probabilidade - Taxa de abstenção por decil', loc='left', fontweight='bold', fontsize=14)
        ax.set_xticks(range(10), ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], color=CINZA1)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.set_xlabel('Decil', labelpad=25, loc='center', color=CINZA1)
        ax.yaxis.set_visible(False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(CINZA9)
        ax.grid(False)
        
        # Annotate absent rate inside each bar with increased font size
        for bar, absent_rate in zip(bars, decile_df['absent_rate']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height - 0.02, f'{absent_rate*100:.1f}%', ha='center', va='top', color='white', fontsize=10.4)
            
    except Exception as e:
        raise CustomException(e, sys)