# Libs needed
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
from scipy import stats
from sklearn import ensemble
from sklearn import metrics
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
rcParams['figure.figsize'] = 25, 10
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # font size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # font size of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # font size of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # font size of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend font size
plt.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

### EDA 
def distribution_plot_by_var(df, var):
    '''
    Plots the distribution of numerical variables
    Pair plot or distribution plot (histogram and scatter)
    '''
    distribution_plot = sns.pairplot(df, hue=var, size=3, diag_kind="kde", diag_kws= {'shade':True}, plot_kws = {'alpha':0.3})

    return distribution_plot

def distribution_plot(df):
    '''
    Plots the distribution of numerical variables
    Pair plot or distribution plot (histogram and scatter)
    '''
    distribution_plot = sns.pairplot(df, size=3, diag_kind="kde", diag_kws= {'shade':True}, plot_kws = {'alpha':0.3})

    return distribution_plot

def joint_plot(df, var1, var2, shade):
    '''
    Creates a joint plot for two numeric variables:
    scatter plot/distribution of the two variables
    r-squared and p-value
    ''' 
    joint = sns.jointplot(var1, var2, data=df, kind='reg', color=shade)

    return joint

def corr_matrix(df):
    '''Generates Correlation Matrix'''
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(15, 10))
    colormap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=colormap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": 1})
 
def histo(df, var, bins):
    '''Generates basic histograms'''
    num_bins = bins
    plt.hist(df[var], num_bins, normed=1, facecolor='blue', alpha=0.5)
    plt.show()

def kde_plot(df, var, title, xlabel):
    ''' Generates a Kernel Density Plot'''
    kde = sns.kdeplot(df[var], shade=True)
    plt.title(title)
    plt.xlabel(xlabel)

def scatter(x, y, title, xlabel, ylabel):
    '''Generates Scatter plot by date'''
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

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
def linear_regression(X, y, cv):
    '''
    Linear regression
    X -> independent variables
    y -> dependent variable
    cv -> number of K-folds for cross-validation 
    '''
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print("Training Set:", X_train.shape, y_train.shape)
    # print("Testing Set:",X_test.shape, y_test.shape)

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