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



def extract_data(url, delete_columns, rows_to_skip):
    

    data = pd.read_excel(url, skiprows=rows_to_skip)
    #data = data.loc[data['Indicator Name'] == indicator]

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
    #df2 = df2.rename(columns={"index":"Year"})
    
    return df_country, df_years

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'cyan'}
    mrk_dic = {0:'*',1:'+',2:'.'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][1], samples[sample][2], 
                    color=colors[sample], marker=markers[sample], s=50)
    plt.scatter(df_centriod[0], df_centriod[1], c='black', marker='x', s=100)
    plt.xlabel(cluster_year[0])
    plt.ylabel(cluster_year[1])
    plt.title('Cluster of Population Growth')
    plt.savefig('clustered_data.png')
    plt.show()
    
    
def centriods(data, axis, clusters):
    return [data.loc[data["label"] == c, axis].mean() for c in range(clusters)]


# define the true objective function
def objective(x, a, b, c, d):
    x = x - 2015.0
    #return a*x + b*x**4 + c*x**3 + d
    return a*x**3 + b*x**2 + c*x + d
    

#---------------------------------INITIALIZATIONS______________________________

cluster_year = ["1975", "2020"]
#url = 'https://api.worldbank.org/v2/en/indicator/AG.LND.ARBL.ZS?downloadformat=excel'
url = 'https://api.worldbank.org/v2/en/indicator/SP.POP.GROW?downloadformat=excel'
columns_to_delete = ['Country Code', 'Indicator Name', 'Indicator Code']
rows_to_skip = 3
year = ['2009', '2010', '2011', '2012', '2013', '2014', 
        '2015', '2016', '2017', '2018', '2019', '2020', '2021']
clusters = 3


#get the required data by calling the extract_data function.
#this gets the entire dataset
df_country, df_years = extract_data(url, columns_to_delete, rows_to_skip)
'''
#extract the data required
df_arable_cluster = df_country.loc[df_country.index, 
                                   ["Country Name", "1975", "2020"]]


#re-assign the data and convert it to a numpay array
x = df_arable_cluster
x = x.dropna().values

plt.figure()
plt.scatter(x[:,1], x[:,2], color='black')
plt.savefig('raw_data.png')
plt.show()


#find the number of clusters for the in use

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x[:,1:2])
    wcss.append(kmeans.inertia_)

plt.figure()    
plt.plot(range(1, 11), wcss)
plt.savefig('cluster-elbow.png')
plt.show()



# Create a model based on set cluster counts
model = KMeans(n_clusters=clusters, init='k-means++', 
               n_init=100, max_iter=1000)

# Fit to the data and predict the cluster assignments for each data point
km_clusters = model.fit_predict(x[:,1:2])

#generate a dataframe that includes the result of the label/classification
df_clustered = df_arable_cluster[cluster_year]
df_clustered = df_clustered.dropna()
df_clustered['label'] = km_clusters.tolist()
print(df_clustered)

#get the x and y coordinates of the centroids
centriod_x = centriods(df_clustered, cluster_year[0], clusters)
centriod_y = centriods(df_clustered, cluster_year[1], clusters)

#I needed to transpose, hence, converted the centroids to a dataframe
n_array = np.array([centriod_x, centriod_y])
df_centriod = pd.DataFrame(n_array)
df_centriod = df_centriod.T
print(df_centriod)

#plot the clusters according to the label and classification
plot_clusters(x, km_clusters)


# countries in each of the clusters
# get a df of the countries based on classification
# select random countries

df_cluster_analysis = df_arable_cluster.loc[:].dropna()
df_cluster_analysis['label'] = km_clusters.tolist()

#df_cluster_analysis.sort_values('classification').tail(20)

#countries in cluster 2: Somalia, Saudi Arabia, Kuwait
#countries in cluster 1: United Kingdom, France, China, Japan
#countries in cluster 0: Nigeria, South Africa, Zambia

#df_cluster_analysis.loc[df_cluster_analysis['classification'] == 2]

countries_2 = ['Somalia', 'Saudi Arabia', 'Kuwait']
countries_1 = ['United Kingdom', 'France', 'China', 'Japan']
countries_0 = ['Nigeria', 'Angola', 'Zambia']

plt.figure()
for country in countries_2:
    plt.plot(df_years['Year'], df_years[country], label=country)
    plt.legend()
plt.show()

plt.figure()
for country in countries_1:
    plt.plot(df_years['Year'], df_years[country], label=country)
    plt.legend()
plt.show()

plt.figure()
for country in countries_0:
    plt.plot(df_years['Year'], df_years[country], label=country)
    plt.legend()
plt.show()



'''
#get dataset
df_fitting = df_years.loc[year, ["Year", "Cameroon"]].apply(pd.to_numeric, 
                                                         errors='coerce')
x = df_fitting.dropna().to_numpy()

#create a variable for the input and the output
x, y = x[:, 0], x[:, 1]

# curve fit
popt, _ = opt.curve_fit(objective, x, y)


# summarize the parameter values
a, b, c, d = popt
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))

param, covar = opt.curve_fit(objective, x, y)

#generate parameters for the error ranges
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


# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color='red')
plt.savefig('curve_fit.png')
plt.show()
plt.plot(x_line, y_line, '--', color='red')
plt.fill_between(x, low, up, alpha=0.7, color='green')
plt.savefig('with error ranges.png')
plt.show()