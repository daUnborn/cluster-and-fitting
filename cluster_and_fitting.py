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
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#--------------------------create functions---------------------------------#

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
    data = pd.read_excel(url, skiprows=3)
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
    data : DATAFRAME
        Dataframe containing all the datapoints of the 2 axis.
    axis : INT
        0 0r 1 which can be either for the row or column.
    clusters : LIST
        clusters of the dataframe.

    Returns
    -------
    list
        List of the mean position of all the datapoints.

    '''
    return [data.loc[data["label"] == c, axis].mean() 
            for c in range(clusters)]

def plot_clusters(x, clusters):
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
    
    plt.scatter(x[clusters == 0, 0], x[clusters == 0, 1], s = 25, 
                label = 'cluster 0')
    plt.scatter(x[clusters == 1, 0], x[clusters == 1, 1], s = 25, 
                label = 'cluster 1')
    plt.scatter(x[clusters == 2, 0], x[clusters == 2, 1], s = 25, 
                label = 'cluster 2')
    
    plt.scatter(df_centriod[0], df_centriod[1], c='black', marker='x', s=100)
    
    #plt.gca().add_patch(Rectangle((400000, 52), 500000, 34, linewidth=1, edgecolor='b', facecolor='none'))
    
    plt.xlabel(end_year)
    plt.ylabel(start_year)
    plt.title('Cluster of Agricultural Land (%)')
    plt.legend()
    plt.savefig('agric_cluster.png')
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


def plot_chart(data, x_data, y_data, kind, xlabel, ylabel, title):
    """
    This function plots a chart based on the kind specified

        Parameters:
            a. data: The dataframe in considration
            b. x_data: data of the x-axis
            c. y_data: data of the y-axis
            d. kind: the kind of chart to plot e.g. bar, line
    
    """
    plt.figure(figsize=(100, 100))
    data.plot(x_data, y_data, kind=kind)
    plt.legend(fontsize=12)
    plt.xticks(rotation='vertical')
    plt.xlabel(xlabel, fontsize='18')
    plt.ylabel(ylabel, fontsize='18')
    plt.title(title, fontsize='18')
    plt.tick_params(axis='both', labelsize=18)
    plt.savefig(title+'.svg', bbox_inches="tight")
    plt.show()

def heat_maps(years, data):
    '''
    Parameters
    ----------
    years : LIST
        A list of years to be considered for correlation.
    years : dataframe
        dataframe to be used for the correlation.

    Returns
    -------
    None.

    '''
    
    d = {years[0]: data[years[0]],  years[1]: data[years[1]]}
    df = pd.DataFrame(data=d)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # This increases the size of the heatmap.
    plt.figure(figsize=(10, 6))
    
    #render the heatmap
    sns.heatmap(df.corr(), annot=True, fmt='.5g')

    # Give a title to the heatmap
    plt.title('Correlation heatmap for '+start_year+'-'+end_year, fontsize=18, pad=(24))

    #save the image
    plt.savefig('Correlation_.png')

    plt.show()

#----------------------variable initialization------------------------------#

#declare the required variables
n_clusters = 3
start_year = '1980'
end_year = '2020'
url = 'https://api.worldbank.org/v2/en/indicator/AG.LND.AGRI.ZS?downloadformat=excel'
columns_to_delete = ['Country Code', 'Indicator Name', 'Indicator Code']

#the years to be targetted for the curve fitting
year = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', 
        '2017', '2018', '2019', '2020']

#year to be predicted. We could not predict the future
predict_year = 2015

mpl.rcParams['figure.dpi'] = 300

#call the extrac_data function and assign the values to 2 variables
df_country, df_years = extract_data(url, columns_to_delete, 
                          indicator='Agricultural land (% of land area)')

#since we do not need all the data, I extract data from 2 different years
#and the countries for our clustering. The countries is required to be able 
#to know which countries fall in each cluster/class/label
df_clustering = df_country.loc[df_country.index, 
                               ["Country Name", start_year, end_year]]

##Convert the clustering data to an array
x = df_clustering[[start_year, end_year]]
x = x.dropna().values
#print(x)

x_data_country='Country Name'
y_data_country=[start_year, end_year]
#----------------------data exploration-------------------------------------#

#descriptive statistics
print(df_country.describe())
print(df_country.info())
print(df_country.isnull())
print(df_country.isnull().sum())

df_country = df_country.dropna()

print(df_country.isnull().sum())

df_clustering[[start_year, end_year]].describe().to_csv('describe.csv')

print(df_clustering[[start_year, end_year]].describe())

#correlation - again, the correlation between the years in review 
#were considered

heat_maps([start_year, end_year], df_clustering[[start_year, end_year]])

#----------------------clustering-------------------------------------------#

#plot a graph of the raw data
plt.figure(figsize=(14, 10))
plt.scatter(x[:,0], x[:,1], label='% of land area', marker='o', s=100)
plt.xlabel(end_year, fontsize=30)
plt.ylabel(start_year, fontsize=30)
plt.title('Data without clustering', fontsize=30)
plt.tick_params(axis='both', labelsize=30)
plt.legend(fontsize=30)
plt.savefig('agric_raw_data.png')
plt.show()

#-------------------------without normatilzation-----------------------------#
#determine the elbow which is to be used as the cluster
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x[:,0:1])
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(18, 11))
plt.plot(range(1, 11), wcss, label='Cluster', marker='o')
plt.xlabel('Number of clusters', fontsize=30)
plt.ylabel('WCSS', fontsize=30)
plt.title('Elbow Method (n_clusters)', fontsize=30)
plt.tick_params(axis='both', labelsize=30)
plt.legend(fontsize=30)
plt.savefig('elbow.png')
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
#print(df_clustered)

centroids = model.cluster_centers_
#print(model.cluster_centers_)

#initiailize the centroids
centriod_x = centriods(df_clustered, start_year, 3)
centriod_y = centriods(df_clustered, end_year, 3)

n_array = np.array([centriod_x, centriod_y])
df_centriod = pd.DataFrame(n_array)
df_centriod = df_centriod.T
#print(df_centriod)

plot_clusters(x, km_clusters)
#plot_cluster_2(n_clusters, km_clusters, centroids)

score = silhouette_score(x, km_clusters, metric='euclidean')

print("Silhouette score: ", score)

#--------------------with normalization-------------------------------------#
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x_scaled[:,0:1])
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(18, 11))
plt.plot(range(1, 11), wcss, label='Cluster', marker='o')
plt.xlabel('Number of clusters', fontsize=30)
plt.ylabel('WCSS', fontsize=30)
plt.title('Elbow Method (n_clusters)', fontsize=30)
plt.tick_params(axis='both', labelsize=30)
plt.legend(fontsize=30)
plt.savefig('elbow.png')
plt.show()

# Create a model based on 2 centroids
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10,
               random_state = 0)

# Fit to the data and predict the cluster assignments for each data point
km_clusters_normalized = kmeans.fit_predict(x_scaled)

#get a list of the countries and their labels so that they can be easily
#analysed
df_clustered_normalized = df_clustering[[start_year, end_year]]
df_clustered_normalized = df_clustered.dropna()
df_clustered_normalized['cluster'] = km_clusters_normalized.tolist()

centroids_normalized = kmeans.cluster_centers_
#print(model.cluster_centers_)

#initiailize the centroids
centriod_x_n = centriods(df_clustered_normalized, start_year, 3)
centriod_y_n = centriods(df_clustered_normalized, end_year, 3)

n_array = np.array([centriod_x_n, centriod_y_n])
df_centriod = pd.DataFrame(n_array)
df_centriod = df_centriod.T
#print(df_centriod)

plot_clusters(x_scaled, km_clusters_normalized)
#plot_cluster_2(n_clusters, km_clusters, centroids)

score = silhouette_score(x_scaled, km_clusters_normalized, metric='euclidean')

print("Silhouette score: ", score)

#------------------end of normalization------------------------------------#

#analysis of the clusters
m = df_country[['Country Name', start_year, end_year]]
m = m.loc[(m[start_year] < 0) & (m[end_year] < 0)]
print(m)

n = df_clustering[['Country Name', start_year, end_year]]
n = n.loc[(n[start_year] < 0) & (n[end_year] > 0)]
print(n)

# countries in each of the clusters
# get a df of the countries based on classification
# select random countries

df_cluster_analysis = df_clustering.loc[:].dropna()
df_cluster_analysis['label'] = km_clusters.tolist()

g = df_cluster_analysis.loc[df_cluster_analysis['label'] == 2]
s = g['1980']
e = g['2020']

g['increase'] = ((e - s)/e) * 100
g['diff'] = (g['1980'] - g['2020'])
print(g.to_string())

print(min(e), max(e))

#2 = 59.92787441663131
#1 = 2.79515069113981
#0 = 85.63899868247694

#cluster 0 countries: Denmark, Australia, United Kingdom
#cluster 1 countries: Japan, Qatar, Gabon
#cluster 2 countries: Argentina, Belarus, Senegal

g = df_cluster_analysis.loc[df_cluster_analysis['label'] == 0]
df_bar = g.loc[g['Country Name'].isin(['Denmark', 'Australia', 'United Kingdom'])]
plot_chart(df_bar, x_data_country,  y_data_country, 'bar', 'Country', '%land', 'Cluster 0 - Agricultural land')

g = df_cluster_analysis.loc[df_cluster_analysis['label'] == 1]
df_bar1 = g.loc[g['Country Name'].isin(['Gabon', 'Japan', 'Singapore'])]
plot_chart(df_bar1, x_data_country,  y_data_country, 'bar', 'Country', '%land', 'Cluster 1 - Agricultural land')

g = df_cluster_analysis.loc[df_cluster_analysis['label'] == 2]
df_bar2 = g.loc[g['Country Name'].isin(['Argentina', 'Kenya', 'Portugal'])]
plot_chart(df_bar2, x_data_country,  y_data_country, 'bar', 'Country', '%land', 'Cluster 2 - Agricultural land')

#------------------------------fitting---------------------------------------#

#get dataset
df_fitting = df_years.loc[year, ["Year", "United Kingdom"]].apply(pd.to_numeric, 
                                                       errors='coerce')
x_array = df_fitting.dropna().to_numpy()

# choose the input and output variables
x_axis, y_axis = x_array[:, 0], x_array[:, 1]

# curve fit
land, _ = opt.curve_fit(objective, x_axis, y_axis)

# summarize the parameter values
a, b, c, d = land
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))

param, covar = opt.curve_fit(objective, x_axis, y_axis)

sigma = np.sqrt(np.diag(covar))
low, up = err.err_ranges(x_axis, objective, land, sigma)

print('*****************', low, up)
print('covar = ', covar)

# plot input vs output
plt.figure()
plt.plot(x_axis, y_axis, label='United Kingdom % land')
plt.xlabel('years')
plt.ylabel('% land')
plt.title('% land by country')
plt.legend()
plt.savefig('United Kingdom.png')
plt.show()

# define a sequence of inputs between the smallest and largest known inputs
x_line = np.arange(min(x_axis), max(x_axis) + 1, 1)

# calculate the output for the range
y_line = objective(x_line, a, b, c, d)

# Use interp1d to create a function for interpolating y values
interp_func = interpolate.interp1d(x_line.flatten(), y_line)

#initialize a list to add the agricultural land use predictions
land_use_percent = [] 
predict = float(format(interp_func(predict_year), '.3f'))
land_use_percent.append(predict)

# print out the value of the prediction
print('prediction', land_use_percent)

# create a line plot for the mapping function
plt.figure()
plt.plot(x_axis, y_axis, label='United Kingdom % land')
plt.plot(x_line, y_line, '--', color='red')
plt.scatter(predict_year, land_use_percent, marker=('x'), s=100, color='red', 
            label=f"pop growth of {predict_year} precipitation is {predict}.")
plt.xlabel('years')
plt.ylabel('% land')
plt.title('% land by country')
plt.legend()
plt.savefig('curve_fit.png')
plt.show()

#create a line plot of the original data, the curve fitting and the
#error ranges
plt.figure()
plt.plot(x_axis, y_axis, label='United Kingdom % land')
plt.plot(x_line, y_line, '--', color='red')
plt.fill_between(x_axis, low, up, alpha=0.7, color='green', label='error range')
plt.xlabel('years')
plt.ylabel('% land')
plt.title('% land by country')
plt.legend()
plt.savefig('with error ranges.png')
plt.show()




