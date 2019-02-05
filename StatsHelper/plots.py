'''
EDA Plots 
Version: 1.0

This module made to do EDA before going into more in-depth analyses.
Creates plots and then saves them to a folder to later use.
'''

# Libs needed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## EDA 
def distribution_plot_by_var(df, var, fig_filename):
    '''
    Plots the distribution of numerical variables
    Pair plot or distribution plot (histogram and scatter)
    '''
    distribution_plot = sns.pairplot(df, hue = var, size = 3, diag_kind = "kde", diag_kws =  {'shade':True}, plot_kws = {'alpha':0.3})
    fig = distribution_plot.fig
    fig.savefig(fig_filename)

    return distribution_plot

def distribution_plot(df, fig_filename):
    '''
    Plots the distribution of numerical variables
    Pair plot or distribution plot (histogram and scatter)
    '''
    distribution_plot = sns.pairplot(df, size = 3, diag_kind = "kde", diag_kws = {'shade':True}, plot_kws = {'alpha':0.3})
    fig = distribution_plot.fig
    fig.savefig(fig_filename)

    return distribution_plot

def joint_plot(df, var1, var2, shade, fig_filename):
    '''
    Creates a joint plot for two numeric variables:
    scatter plot/distribution of the two variables
    r-squared and p-value
    ''' 
    joint = sns.jointplot(var1, var2, data = df, kind = 'reg', color = shade)
    fig = joint.fig
    fig.savefig(fig_filename)

    return joint

def corr_matrix(df, fig_filename):
    '''Generates Correlation Matrix'''
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize = (15, 10))
    colormap = sns.diverging_palette(240, 10, n = 9, as_cmap = True)
    sns.heatmap(corr, mask = mask, cmap = colormap, center = 0,
                square = True, linewidths = .5, cbar_kws = {"shrink": 1})
    f.savefig(fig_filename)
 
def histogram(data, n_bins, cumulative = False, x_label, y_label, title, color, fig_filename):
    '''
    Generates a histogram 
    data <- the df
    n_bins <- how many discrete bins we want for our histogram
    cumulative <- a boolean which allows us to select whether our histogram is cumulative or not
    x_label <- the x-label 
    y_label <- the y-label 
    title <- the title of the plot
    fig_filename <- the name of the file that you want to save the figure as
    '''
    # Create the plot object
    _, ax = plt.subplots()
    # Plot the data
    ax.hist(data, n_bins = n_bins, cumulative = cumulative, color = color)
    # Label the axes and provide a title
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    # Save the fig
    plt.savefig(fig_filename)

def overlaid_histogram(data1, data2, n_bins = 0, data1_name, data1_color, data2_name, data2_color, x_label, y_label, title, fig_filename):
    '''
    Generates a overlaid histogram (2 variables)
    data1 <- the first variable 
    data2 <- the first variable 
    n_bins <- how many discrete bins we want for our histogram
    data1_color <- the color of variable 1
    data2_color <- the color of variable 2
    data1_name <- label the graph with variable 1's name 
    data2_name <- label the graph with variable 2's name
    x_label <- the x-label 
    y_label <- the y-label 
    title <- the title of the plot
    fig_filename <- the name of the file that you want to save the figure as
    '''
    # Set the bounds for the bins so that the two distributions are fairly compared
    max_nbins = 10
    data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
    binwidth = (data_range[1] - data_range[0]) / max_nbins
    # If the bins are 0, create bins, else bins are 0
    if n_bins == 0
        bins = np.arange(data_range[0], data_range[1] + binwidth, binwidth)
    else: 
        bins = n_bins
    # Create the plot
    _, ax = plt.subplots()
    ax.hist(data1, bins = bins, color = data1_color, alpha = 1, label = data1_name)
    ax.hist(data2, bins = bins, color = data2_color, alpha = 0.75, label = data2_name)
    # Label the axis's
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'best')
    # Save the fig
    plt.savefig(fig_filename)

def kde_plot(df, var, title, xlabel, fig_filename):
    ''' Generates a Kernel Density Plot'''
    kde = sns.kdeplot(df[var], shade=True)
    plt.title(title)
    plt.xlabel(xlabel)
    fig = kde.get_figure()
    fig.savefig(fig_filename)

def scatterplot(x_data, y_data, x_label, y_label, title, color, s, alpha, yscale_log = False, fig_filename):
    ''' 
    Generates a scatter plot 
    x_data <- first variable 
    y_data <- second variable 
    x_label <- the x-label 
    y_label <- the y-label 
    title <- the title of the plot
    color <- the color of the dots
    s <- the size of the dots 
    alpha <- the alpha 
    yscale_log <- scales the y-axis (True if want to scale and False is not scaling)
    fig_filename <- the name of the file that you want to save the figure as 
    '''
    # Create the plot object
    _ , ax = plt.subplots()
    # Plot the data
    ax.scatter(x_data, y_data, s = s, color = color, alpha = alpha)
    # If yscale_log is true then scale the y-axis
    if yscale_log == True:
        ax.set_yscale('log')
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Save the fig
    plt.savefig(fig_filename)

def lineplot(x_data, y_data, x_label, y_label, title, color, lw, alpha, fig_filename):
    ''' 
    Generates a line plot
    x_data <- first variable 
    y_data <- second variable 
    x_label <- the x-label 
    y_label <- the y-label 
    title <- the title of the plot
    color <- the color of the dots
    lw <- the line width 
    alpha <- the alpha 
    fig_filename <- the name of the file that you want to save the figure as 
    '''
    # Create the plot object
    _ , ax = plt.subplots()
    # Plot the best fit line, set the line-width (lw), color and
    # transparency (alpha) of the line
    ax.plot(x_data, 
            y_data, 
            lw = lw, 
            color = color, 
            alpha = alpha)
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Save the fig
    plt.savefig(fig_filename)

def boxplot(x_data, y_data, base_color, median_color, x_label, y_label, title, fig_filename):
   ''' 
    Generates a box plot
    x_data <- variables for box 
    y_data <- values for the boxes 
    x_label <- the x-label 
    y_label <- the y-label 
    title <- the title of the plot
    base_color <- the color of the base plot
    median_color <- the color of the median bar
    fig_filename <- the name of the file that you want to save the figure as 
    '''
   # Create the plot object
    _ , ax = plt.subplots()
    # Draw box-plots, specifying desired style
    ax.boxplot(y_data,
               patch_artist = True,
               medianprops = {'color': median_color},
               boxprops = {'color': base_color'facecolor': base_color},
               whiskerprops = {'color': base_color},
               capprops = {'color': base_color})

    # By default, the tick label starts at 1 and increments by 1 for
    # each box drawn. This sets the labels to the ones we want
    ax.set_xticklabels(x_data)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    # Save the fig
    plt.savefig(fig_filename)
