#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 12:31:18 2022

@author: camaike
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import err_ranges as err
from sklearn.cluster import KMeans
from scipy import interpolate

def extract_data(url, delete_columns, indicator):
    '''
    

    Parameters
    ----------
    url : STRING
        A string of the URL of the data set. It could be a web URL 
        or file path.
    delete_columns : LIST
        List of columns we want to drop from the data set.
    rows_to_skip : TYPE
        Number of row to skip.

    Returns
    -------
    df_country : dataframe
        dataframe of years as columns.
    df_years : dataframe
        dataframe of countries as columns with years as rows.

    '''
    
    #read the data from file and put it in a dataframe
    data = pd.read_csv(url, skiprows=3)
    data = data.loc[data['Indicator Name'] == indicator]

    #dropping columns that are not needed. Also, rows with NA were dropped
    data = data.drop(delete_columns, axis=1)

    #this extracts a dataframe with countries as column
    df_country = data
    
    #this section extract a dataframe with year as columns
    df_years = data.transpose()
    
    #removed the original headers after a transpose and dropped the row
    #used as a header
    df_years = df_years.rename(columns=df_years.iloc[0])
    df_years = df_years.drop(index=df_years.index[0], axis=0)
    df_years['Year'] = df_years.index
    #df_years = df_years.rename(columns={"index":"Year"})
    
    return df_country, df_years

def centriods(data, axis, clusters):
    '''
    Used this function to generate centroids. I checked it out with the 
    regular kmeans.cluster_centroids_ and I got the same values.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    axis : TYPE
        DESCRIPTION.
    clusters : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    '''
    return [data.loc[data["label"] == c, axis].mean() 
            for c in range(clusters)]

def plot_clusters(samples, clusters):
    '''
    One of the methods used to plot the clusters

    Parameters
    ----------
    samples : array
        an arrary of the data to be used for the clustering.
    clusters : array
        an array of the custers.

    Returns
    -------
    None.

    '''
    col_dic = {0:'blue',1:'green',2:'cyan'}
    mrk_dic = {0:'*',1:'+',2:'.'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], 
                    color = colors[sample], marker=markers[sample], s=50)
    plt.scatter(df_centriod[0], df_centriod[1], c='black', marker='x', s=100)
    plt.xlabel(start_year)
    plt.ylabel(end_year)
    plt.title('Cluster of Population Growth')
    plt.savefig('clustered_data.png')
    plt.show()

'''
def plot_cluster_2(n_clusters, km_clusters, centroids):
    
    plt.figure(figsize=(10.0, 10.0))
    col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    
    for l in range(n_clusters): # loop over the different labels
        plt.plot(x[km_clusters==l], y[km_clusters==l], "o", markersize=4, 
                 color=col[l])
        
    # show cluster centres
    for ic in range(n_clusters):
        xc, yc = centroids[ic,:]
        
    plt.plot(xc, yc, "dk", markersize=10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
'''

# define the true objective function
def objective(x, a, b, c, d):
    '''
        This is the model that would be used to determine the line of best fit
        
    '''
    x = x - 2010.0
    return a*x**3 + b*x**2 + c*x + d

#declare the required variables
n_clusters = 3
start_year = '1980'
end_year = '2021'
url = 'API_19_DS2_en_csv_v2_4700503.csv'
columns_to_delete = ['Country Code', 'Indicator Name', 'Indicator Code']
year = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', 
        '2017', '2018', '2019', '2020', '2021']

#year to be predicted
predict_year = 2015
#call the extrac_data function and assign the values to 2 variables
df_country, df_years = extract_data(url, columns_to_delete, 
                          indicator='Population growth (annual %)')

#since we do not need all the data, I extract data from 2 different years
#and the countries for our clustering. The countries is required to be able 
#to know which countries fall in each cluster/class/label
df_clustering = df_country.loc[df_country.index, 
                               ["Country Name", start_year, end_year]]

##Convert the clustering data to an array
x = df_clustering[[start_year, end_year]]
x = x.dropna().values
plt.figure()
plt.scatter(x[:,0], x[:,1])
plt.savefig('raw_data.png')
plt.show()

#determine the elbow which is to be used as the cluster
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x[:,0:1])
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.savefig('cluster.png')
plt.show()

# Create a model based on 2 centroids
model = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10,
               random_state = 0)

# Fit to the data and predict the cluster assignments for each data point
km_clusters = model.fit_predict(x)

#get a list of the countries and their labels so that they can be easily
#analysed
df_clustered = df_clustering[[start_year, end_year]]
df_clustered = df_clustered.dropna()
df_clustered['label'] = km_clusters.tolist()
print(df_clustered)

centroids = model.cluster_centers_
print(model.cluster_centers_)

#initiailize the centroids
centriod_x = centriods(df_clustered, start_year, 3)
centriod_y = centriods(df_clustered, end_year, 3)

n_array = np.array([centriod_x, centriod_y])
df_centriod = pd.DataFrame(n_array)
df_centriod = df_centriod.T
print(df_centriod)

plot_clusters(x, km_clusters)
#plot_cluster_2(n_clusters, km_clusters, centroids)

#get dataset
df_fitting = df_years.loc[year, ["Year", "China"]].apply(pd.to_numeric, 
                                                       errors='coerce')
x = df_fitting.dropna().to_numpy()

# choose the input and output variables
x, y = x[:, 0], x[:, 1]

# curve fit
popt, _ = opt.curve_fit(objective, x, y)

# summarize the parameter values
a, b, c, d = popt
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))

param, covar = opt.curve_fit(objective, x, y)

sigma = np.sqrt(np.diag(covar))
low, up = err.err_ranges(x, objective, popt, sigma)

print(low, up)
print('covar = ', covar)

# plot input vs output
plt.plot(x, y)

# define a sequence of inputs between the smallest and largest known inputs
x_line = np.arange(min(x), max(x) + 1, 1)

# calculate the output for the range
y_line = objective(x_line, a, b, c, d)

# Use interp1d to create a function for interpolating y values
interp_func = interpolate.interp1d(x_line.flatten(), y_line)

#initialize a list to add the population growth predictions
prediction = [] 
predict = float(format(interp_func(predict_year), '.3f'))
prediction.append(predict)

#print out the value of the prediction
print(prediction)

# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color='red')
plt.scatter(predict_year, prediction, marker=('x'), s=100, color='red', 
            label=f"pop growth of {predict_year} precipitation is {predict}.")
plt.savefig('curve_fit.png')
plt.show()

#create a line plot of the original data, the curve fitting and the
#error ranges
plt.figure()
plt.plot(x, y)
plt.plot(x_line, y_line, '--', color='red')
plt.fill_between(x, low, up, alpha=0.7, color='green')
plt.savefig('with error ranges.png')
plt.show()