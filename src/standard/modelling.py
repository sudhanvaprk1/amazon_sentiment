from scipy.sparse import load_npz
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_features():
    """
    Loads engineered features
    """
    X_train = load_npz("data/train_tfidf.npz")
    X_test = load_npz("data/test_tfidf.npz")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")

    return X_train, X_test, y_train, y_test

def tune_model(x_train, y_train):
    
    # Defining the model
    model = XGBClassifier(n_jobs=-1)
    
    # Defining parameters dictionary for hyperparameter tuning
    parameters = {
                    "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
                    "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4],
                 }

    # Applying GridSearchCV 
    clf = GridSearchCV(model, param_grid=parameters, cv=5, scoring='roc_auc', verbose=2, return_train_score=True, n_jobs=-1)
    clf.fit(x_train, y_train)
    results = pd.DataFrame.from_dict(clf.cv_results_)
    
    # Train AUC scores
    train_auc_score = results['mean_train_score']
    train_auc_std = results['std_train_score']

    # CV AUC scores
    cv_auc_score = results['mean_test_score']
    cv_auc_std = results['std_test_score']

    # Best Hyperparameteres
    optimal_learning_rate = clf.best_params_['learning_rate']
    optimal_gamma = clf.best_params_['gamma']
    best_score = clf.best_score_

    print("Optimal learning rate: ", optimal_learning_rate)
    print("Optimal Gamma: ", optimal_gamma)
    print("Best Score: ", best_score)
    print("="*100)
    
    # Plotting Confusion matrix for train and cv
    max_scores = pd.DataFrame(clf.cv_results_).groupby(['param_learning_rate', 'param_gamma']).max().unstack()[['mean_test_score', 'mean_train_score']]
    fig, ax = plt.subplots(1,2, figsize=(30,6))

    sns.heatmap(max_scores.mean_train_score, annot = True, fmt='.4g', ax=ax[0],annot_kws={"size": 30})
    sns.heatmap(max_scores.mean_test_score, annot = True, fmt='.4g', ax=ax[1],annot_kws={"size": 30},cmap="YlGnBu")

    ax[0].set_ylim([-0.5, 6])
    ax[0].set_title('Train Set', fontsize = 30)
    ax[0].set_xlabel("Learning Rate", fontsize=25)
    ax[0].set_ylabel("Gamma", fontsize=25)

    ax[1].set_ylim([-0.5, 5])
    ax[1].set_title('CV Set', fontsize = 30)
    ax[1].set_ylabel("Gamma", fontsize=25)
    ax[1].set_xlabel("Learning Rate", fontsize=25)
    plt.show()
    return optimal_learning_rate, optimal_gamma

def predict_using_optimal_params(optimal_learning_rate, optimal_gamma, x_train, y_train, x_test, y_test):
    
    # Training model on Train data with optimal hyperparameters
    model = XGBClassifier(n_jobs=-1, optimal_learning_rate=optimal_learning_rate, gamma=optimal_gamma)
    model.fit(x_train, y_train)

    # Checking predictions on train and test data
    y_pred_train = model.predict_proba(x_train)[:,1]
    train_predictions = model.predict(x_train)
    y_pred_test = model.predict_proba(x_test)[:,1]
    test_predictions = model.predict(x_test)

    # calculating roc on train and test data
    train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_pred_train)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_pred_test)

    # Plotting ROC AUC curve and confusion matrix
    ax = plt.subplot()
    auc_train=auc(train_fpr, train_tpr)
    auc_test=auc(test_fpr, test_tpr)

    ax.plot(train_fpr, train_tpr, label="Train AUC ="+str(auc_train))
    ax.plot(test_fpr, test_tpr, label="Test AUC ="+str(auc_test))
    plt.legend()
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("AUC")
    plt.grid(b=True, which='major', color='k', linestyle=':')
    ax.set_facecolor("white")
    plt.savefig("artifacts/FPR_vs_TPR.png")
    plt.show()
    print("="*100)
    from sklearn.metrics import confusion_matrix
    print("Confusion Matrix on test data")
    cf_test = confusion_matrix(y_test, model.predict(x_test))
    ax = sns.heatmap(np.array(cf_test), annot=True, fmt="d", cmap='Blues')
    ax.set_title('Confusion Matrix on Test set', fontsize = 25)
    ax.set_xlabel("Actual", fontsize=20)
    ax.set_ylabel("Predicted", fontsize=20)
    sns.set(font_scale=1.4)
    ax.set_ylim(-0.5, 2.5)
    plt.savefig("artifacts/confusion_matrix.png")
    plt.show()
    
    # returning test predictions which we will require to calculate false positives
    return model, train_predictions, test_predictions

def train_model(tune_model=False):

    # Loading feature engineered data
    X_train, X_test, y_train, y_test = load_features()

    # Training the model on featurized data
    if tune_model:
        optimal_learning_rate, optimal_gamma = tune_model(X_train, y_train)
    else:
        from properties import get_config
        config = get_config()
        print(config)
        optimal_learning_rate, optimal_gamma = config['modelling']['optimal_learning_rate'], config['modelling']['optimal_gamma'] 


    # Checking the model performance on unseen data
    model, train_predictions, test_predictions = predict_using_optimal_params(optimal_learning_rate, optimal_gamma, X_train, y_train, X_test, y_test)

    # Calculate metrics 
    metrics = {}

    # train metrics
    metrics['train_accuracy'] = accuracy_score(y_train, train_predictions)
    metrics['train_recall'] = recall_score(y_train, train_predictions)
    metrics['train_precision'] = precision_score(y_train, train_predictions)
    metrics['train_f1'] = f1_score(y_train, train_predictions)

    # test metrics
    metrics['test_accuracy'] = accuracy_score(y_test, test_predictions)
    metrics['test_recall'] = recall_score(y_test, test_predictions)
    metrics['test_precision'] = precision_score(y_test, test_predictions)
    metrics['test_f1'] = f1_score(y_test, test_predictions)

    return model, train_predictions, test_predictions, metrics

if __name__ == "__main__":
    print(train_model())