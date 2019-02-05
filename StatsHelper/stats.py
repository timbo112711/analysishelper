# Libs needed
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pylab import rcParams
import random, string, pickle
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn import ensemble, metrics, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, classification_report, roc_auc_score, roc_curve

class pickleItYo:
    '''Takes two dictionary objects and combines them into one pickle object'''

    def __init__(self, coff_dict, perform_dict):
        self.coff_dict = coff_dict
        self.perform_dict = perform_dict

    def combine_pickels(self, filename):
        ''' 
        Combines the two dicts into one pickle object.
        Assigns a random string of letter and numbers in-front of the actual filename
            ex. hfj8w_model_results.pkl

        filename -> the name of the file that the pickle object is saved to
        '''
        model_file_name = ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(2, 8)) + '_' + filename

        pickle_dict = self.coff_dict.copy()
        pickle_dict.update(self.perform_dict)

        output = open(filename, 'wb')
        final_pickle = pickle.dump(pickle_dict, open(model_file_name, "wb"))
        
        return final_pickle

### t-tests
def assumptions(var, x, y):
    ''' 
    Testing the assumptions of t-tests
    1. Homogeneity of variances 
    2. Normality
    3. Histograms and Q-Q plots
    '''
    print("Homogeneity of Variances")
    t, p = stats.levene(x, y)

    print("Statistic = ", t)
    print("p-value = ", p)

    if p < 0.05:
        print("----------------------")
        print("There is homogeneity of variances and we can proceed")
    else:
        print("----------------------")
        print("If sample size is not equal, use Welch-Satterthwaite")

    print("")
    print("Normality Check")
    w, p = stats.shapiro(x)

    print("W test statistic = ", w)
    print("p-value = ", p)

    if p > 0.05:
        print("----------------------")
        print("We have Normality")
    else:
        print("----------------------")
        print("Normality is violated here!")

    hist = var.plot(kind="hist", bins=10)
    plt.title('Histogram Plot')
    plt.xlabel('Distribution of the Observations')
    plt.show(hist)
    qq_plot = stats.probplot(var, plot= plt)
    plt.show(qq_plot)

def unpaired_ttest(x, y):
    ''' 
    x -> sample 1
    y-> sample 2
    '''
    dof = (x.var() / x.size + y.var() / y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var() / y.size)**2 / (y.size-1))
   
    t, p = stats.ttest_ind(x, y)
    
    print("Independent t-test = ", t)
    print("p-value = ", p)
    print("Welch-Satterthwaite Degrees of Freedom = ", dof)

    if p < 0.05:
        print("----------------------")
        print("Reject the null yo")
    else:
        print("----------------------")
        print("AHHH the means are the same!!!!!")

def paired_ttest(x, y):
    ''' 
    x -> sample 1
    y-> sample 2
    '''
    dof = (x.var() / x.size + y.var() / y.size)**2 / ((x.var() / x.size)**2 / (x.size-1) + (y.var() / y.size)**2 / (y.size-1))
   
    t, p = stats.ttest_rel(x, y)
    
    print("Paired t-test= ", t)
    print("p-value = ", p)
    print("Welch-Satterthwaite Degrees of Freedom = ", dof)

    if p < 0.05:
        print("----------------------")
        print("Reject the null yo")
    else:
        print("----------------------")
        print("AHHH the means are the same!!!!!")

def welch_ttest(x, y): 
    '''
    Welch-Satterthwaite
    Use when there is unequal sample n and unequal variances
    '''
    dof = (x.var() / x.size + y.var() / y.size)**2 / ((x.var() / x.size)**2 / (x.size-1) + (y.var() / y.size)**2 / (y.size-1))
   
    t, p = stats.ttest_ind(x, y, equal_var = False)
    
    print("Welch's t-test = ", t)
    print("p-value = ", p)
    print("Welch-Satterthwaite Degrees of Freedom = ", dof)

    if p < 0.05:
        print("----------------------")
        print("Reject the null yo")
    else:
        print("----------------------")
        print("AHHH the means are the same!!!!!")

def welch_ttest_adjusted_pval(alpha, x, y): 
    ''' 
    Welch-Satterthwaite w/ adjusted p-value
    Use after either an ANOVA or Kruskal-Wallis for post-hoc testing
    '''
    dof = (x.var() / x.size + y.var() / y.size)**2 / ((x.var() / x.size)**2 / (x.size-1) + (y.var() / y.size)**2 / (y.size-1))
   
    t, p = stats.ttest_ind(x, y, equal_var = False)
    
    print("Welch's t-test with adjusted p-vale = ", t)
    print("p-value = ", p)
    print("Welch-Satterthwaite Degrees of Freedom = ", dof)

    if p < alpha:
        print("----------------------")
        print("Reject the null yo")
    else:
        print("----------------------")
        print("AHHH the means are the same!!!!!")

### Regressions
def linear_regression(X, y, combined_tables, res, qq, preds):
    '''
    Linear regressionL split the data, fit a model, get predictions and model stats
    X -> independent variables
    y -> dependent variable
    combined_tables -> name of file to save the combined two tables to (.pkl)
    res -> filename if the residual plot (.png)
    qq -> filename of the Q-Q plot (.png)
    preds -> filename of the predictions plot (.png)
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit a model
    lm = linear_model.LinearRegression()
    model = lm.fit(X_train, y_train)
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X_test)
    # Actual - prediction = residuals
    residuals = y_test - predictions
    # VIF 
    rsquared = r2_score(y_test, predictions)
    VIF = 1/(1-(rsquared))
    # MSE & MAE
    MSE = mean_squared_error(y_test, predictions)
    MAE = mean_absolute_error(y_test, predictions)

    # Create model statistics dataframe
    model_stats = pd.DataFrame({"Constant":np.ones(len(X_test))}).join(pd.DataFrame(X_test.reset_index(drop=True)))
    var = MSE * (np.linalg.inv(np.dot(model_stats.T,model_stats)).diagonal())
    standard_error = np.sqrt(var)
    t_stat = params/ standard_error
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(model_stats)-len(model_stats.columns)-1))) for i in t_stat]
    # Round the stats off to thousandths 
    standard_error = np.round(standard_error,3)
    t_stat = np.round(t_stat,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    # Populate the model stats dataframe
    model_stats_df = pd.DataFrame()
    model_stats_df["Coefficients"],model_stats_df["Standard Errors"],model_stats_df["t-values"],model_stats_df["p-values"] = [params,standard_error,t_stat,p_values]
    model_stats_dict = model_stats_df.to_dict()

    # Populate a second table for VIF, r-squared, MAE, and MSE
    model_performance_df = pd.DataFrame()
    model_performance = pd.DataFrame({"VIF":VIF, "R-Squared":rsquared, "MAE":MAE, "MSE":MSE}, index=[0])
    model_performance_df = model_performance_df.append(model_performance)
    model_performance_dict = model_performance_df.to_dict()

    print(model_stats_df)
    print("")
    print(model_performance_df)

    # Combine the two pickled tables 
    conbined_pickles = pickleItYo(model_stats_dict, model_performance_dict)
    pickles = conbined_pickles.combine_pickels(combined_tables)

    # Plot the residuals 
    fig1 = plt.figure()
    plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train, c='b', s=40, alpha=0.5)
    plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test, c='g', s=40)
    plt.hlines(y=0, xmin=0, xmax=100)
    plt.title("Residual Plot Using Training (blue) and Testing (green) Data")
    plt.ylabel("Residuals")
    plt.xlabel("Predicted Value")
    fig1.savefig(res)
    # Q-Q Plot
    fig2 = plt.figure()
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("QQ-Plot of the Residuals")
    plt.ylabel("Ordered Values")
    plt.xlabel("Quantiles")
    fig2.savefig(qq)
    # Plot the predictions
    fig3 = plt.figure()
    plt.scatter(y_test, predictions, marker='^', c='b')
    plt.title("Regression Model")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    fig3.savefig(preds)

def LASSO_regression(X, y, alpha, cv):
    '''
    LASSO regression
    X -> independent variables
    y -> dependent variable
    alpha -> constant that multiplies the L1 term
    cv -> number of K-folds for cross-validation 
    '''
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print("Training Set:", X_train.shape, y_train.shape)
    # print("Testing Set:",X_test.shape, y_test.shape)

    # Fit a model
    lm = linear_model.Lasso(alpha)
    model = lm.fit(X_train, y_train)
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X_test)

    # Actual - prediction = residuals
    residuals = y_test - predictions

    # VIF 
    rsquared = r2_score(y_test, predictions)
    VIF = 1/(1-(rsquared))

    # MSE & MAE
    MSE = mean_squared_error(y_test, predictions)
    MAE = mean_absolute_error(y_test, predictions)

    # Create model statistics dataframe
    model_stats = pd.DataFrame({"Constant":np.ones(len(X_test))}).join(pd.DataFrame(X_test.reset_index(drop=True)))
    var = MSE * (np.linalg.inv(np.dot(model_stats.T,model_stats)).diagonal())
    standard_error = np.sqrt(var)
    t_stat = params/ standard_error
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(model_stats)-len(model_stats.columns)-1))) for i in t_stat]
    # Round the stats off to thousandths 
    standard_error = np.round(standard_error,3)
    t_stat = np.round(t_stat,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    # Populate the model stats dataframe
    model_stats_df = pd.DataFrame()
    model_stats_df["Coefficients"],model_stats_df["Standard Errors"],model_stats_df["t-values"],model_stats_df["p-values"] = [params,standard_error,t_stat,p_values]
    print(model_stats_df)
    print("")
    print("Variance Inflation Factor: ", VIF)
    print("R-Squared:", rsquared)
    print("MAE: ", MAE)
    print("MSE: ", MSE)
    # print("Model Coefficients:", model.coef_)

    # Plot the residuals 
    plt.figure()
    plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train, c='b', s=40, alpha=0.5)
    plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test, c='g', s=40)
    plt.hlines(y=0, xmin=0, xmax=100)
    plt.title("Residual Plot Using Training (blue) and Testing (green) Data")
    plt.ylabel("Residuals")
    plt.xlabel("Predicted Value")

    plt.figure()
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("QQ-Plot of the Residuals")
    plt.ylabel("Ordered Values")
    plt.xlabel("Quantiles")

    # Plot the predictions
    plt.figure()
    plt.scatter(y_test, predictions, marker='^', c='b')
    plt.title("Regression Model")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")

    ## Cross-Validation 
    scores = cross_val_score(model, X_test, y_test, cv=cv)
    mean_score = np.mean(scores)
    print("Average R-Squared Cross-validated:", mean_score)

    # Plot the CV predictions
    predictions = cross_val_predict(model, X_test, y_test, cv=cv)
    plt.figure()    
    plt.scatter(y_test, predictions, marker='s', c='g')
    plt.title("Regression Model With K-fold Cross-Validation")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")

def ElasticNet_regression(X, y, alpha, cv):
    '''
    ElasticNet regression (Hybrid LASSO)
    X -> independent variables
    y -> dependent variable
    alpha -> constant that multiplies the penalty terms. Defaults to 1.0
    cv -> number of K-folds for cross-validation 
    '''
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print("Training Set:", X_train.shape, y_train.shape)
    # print("Testing Set:",X_test.shape, y_test.shape)

    # Fit a model
    lm = linear_model.ElasticNet(alpha)
    model = lm.fit(X_train, y_train)
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X_test)

    # Actual - prediction = residuals
    residuals = y_test - predictions

    # VIF 
    rsquared = r2_score(y_test, predictions)
    VIF = 1/(1-(rsquared))

    # MSE & MAE
    MSE = mean_squared_error(y_test, predictions)
    MAE = mean_absolute_error(y_test, predictions)

    # Create model statistics dataframe
    model_stats = pd.DataFrame({"Constant":np.ones(len(X_test))}).join(pd.DataFrame(X_test.reset_index(drop=True)))
    var = MSE * (np.linalg.inv(np.dot(model_stats.T,model_stats)).diagonal())
    standard_error = np.sqrt(var)
    t_stat = params/ standard_error
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(model_stats)-len(model_stats.columns)-1))) for i in t_stat]
    # Round the stats off to thousandths 
    standard_error = np.round(standard_error,3)
    t_stat = np.round(t_stat,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    # Populate the model stats dataframe
    model_stats_df = pd.DataFrame()
    model_stats_df["Coefficients"],model_stats_df["Standard Errors"],model_stats_df["t-values"],model_stats_df["p-values"] = [params,standard_error,t_stat,p_values]
    print(model_stats_df)
    print("")
    print("Variance Inflation Factor: ", VIF)
    print("R-Squared:", rsquared)
    print("MAE: ", MAE)
    print("MSE: ", MSE)
    # print("Model Coefficients:", model.coef_)
    
    # Plot the residuals 
    plt.figure()
    plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train, c='b', s=40, alpha=0.5)
    plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test, c='g', s=40)
    plt.hlines(y=0, xmin=0, xmax=100)
    plt.title("Residual Plot Using Training (blue) and Testing (green) Data")
    plt.ylabel("Residuals")
    plt.xlabel("Predicted Value")

    plt.figure()
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("QQ-Plot of the Residuals")
    plt.ylabel("Ordered Values")
    plt.xlabel("Quantiles")

    # Plot the predictions
    plt.figure()
    plt.scatter(y_test, predictions, marker='^', c='b')
    plt.title("Regression Model")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")

    ## Cross-Validation 
    scores = cross_val_score(model, X_test, y_test, cv=cv)
    mean_score = np.mean(scores)
    print("Average R-Squared Cross-validated:", mean_score)

    # Plot the CV predictions
    predictions = cross_val_predict(model, X_test, y_test, cv=cv)
    plt.figure()
    plt.scatter(y_test, predictions, marker='s', c='g')
    plt.title("Regression Model With K-fold Cross-Validation")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")

def gradient_boosting(X, y, n_estimators, criterion, learning_rate, max_depth, random_state, loss):
    '''
    Gradient Boosting Regressor
    X -> independent variables
    y -> dependent variable
    n_estimators -> number of estimators
    criterion -> The function to measure the quality of a split. 
                “friedman_mse” for the mean squared error with improvement score by Friedman, 
                “mse” for mean squared error, and
                “mae” for the mean absolute error.
    learning_rate -> float, optional (default=0.1). Learning rate shrinks the contribution of each tree by learning_rate
    max_depth -> maximum depth of the individual regression estimators
    random_state ->
    loss -> ‘ls’:refers to least squares regression, 
            ‘lad’:(least absolute deviation) is a highly robust loss function solely based on order information of the input variables, 
            ‘huber’:combination of 'ls' and 'lad',
            ‘quantile: allows quantile regression (use alpha to specify the quantile),
            optional (default=’ls’)
    '''
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    params = {'n_estimators': n_estimators, 'criterion':criterion, 'max_depth': max_depth,
              'learning_rate': learning_rate, 'loss': loss}

    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    MSE = mean_squared_error(y_test, model.predict(X_test))
    print("MSE: %.4f" % mse)

    # Plot training deviance
    # Compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    # Plot feature importance
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, boston.feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    
def logit_feature_selection(X, y):
    '''
    Perform feature selection for logistic regression using SMOTE and RFE.
    SMOTE algorithm(Synthetic Minority Oversampling Technique). 
        Works by creating synthetic samples from the minor class instead of creating copies,
        Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observations.
    Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model and choose either the best or 
        worst performing feature, setting the feature aside and then repeating the process with the rest of the features

    Returns the best features

    X -> Independent variables 
    y -> Dependent variable (binary)
    '''
    os = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    columns = X_train.columns
    os_data_X,os_data_y=os.fit_sample(X_train, y_train)
    os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
    os_data_y= pd.DataFrame(data=os_data_y,columns=y)

    # Check the numbers of our data
    print("length of oversampled data is ",len(os_data_X))
    print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
    print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
    print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
    print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

    logit = LogisticRegression()

    # Feature selection 
    rfe = RFE(logit, 20)
    rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
    print(rfe.support_)
    print(rfe.ranking_)

def logit_regression(X, y):
    '''
    Perform logistic regression

    X -> Feature selection variables 
    y -> Dependent variable (binary)
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    logit = LogisticRegression()
    logit.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    confusion_matrix = confusion_matrix(y_test, y_pred)
    classification_report = classification_report(y_test, y_pred)

    # Convert the classification report and confusion matrix to a df 
    classification_report_df = pd.DataFrame(classification_report)
    classification_report_df_dict = classification_report_df_dict.to_dict()
    confusion_matrix_df = pd.DataFrame(confusion_matrix)
    confusion_matrix_df_dict = confusion_matrix_df_dict.to_dict()

    # Combine the classification report and confusion matrix 
    conbined_pickles = pickleItYo(classification_report_df_dict, confusion_matrix_df_dict)
    pickles = conbined_pickles.combine_pickels(combined_tables)

    # ROC Curve
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    print("")
    print(confusion_matrix)
    print("")
    print(classification_report)

## Clustering 
def plot_clusters(data, algorithm, args, kwds):
    '''
    Clustering function that can use any type of clustering algorithm
    data <- input data
    algorithm <- KMeans, AffinityPropagation, SpectralClustering, AgglomerativeClustering, DBSCAN
    args <- bandwidth value, etc.
    kwds <- 'cluster_all':False, or {'n_clusters':6} or {'n_clusters':6, 'linkage':'ward'}
    example <- plot_clusters(data, cluster.KMeans, (), {'n_clusters':6})
    '''
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
