import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def getBenchmarkGroup(hospitalId, groupSize):
    
    continuous = pd.read_excel('continuousFeatures.xlsx').sort_values('HOSP_NIS')
    categorical = pd.read_excel('categorical.xlsx').sort_values('HOSP_NIS')
    
    continuousNormalize = StandardScaler().fit_transform(continuous.drop(columns=['HOSP_NIS']))
    featureArray = np.append(continuousNormalize,
                             categorical.drop(columns=['HOSP_NIS']).to_numpy(), axis=1)
    X_pca = PCA(n_components = 11).fit_transform(featureArray)
    
    cols = [f'PCA{i}' for i in range(X_pca.shape[1])]
    featuresDf = pd.DataFrame(columns=cols, data=X_pca).fillna(0)
    featuresDf['Hospital'] = continuous['HOSP_NIS']

    #get focus hospital's array of feature values
    knnX = featureArray[featuresDf.loc[featuresDf['Hospital'] == hospitalId].index[0]] 

    #input knnX array to find nearest neighbors
    neigh = NearestNeighbors(n_neighbors=groupSize).fit(featureArray)
    _, ind = neigh.kneighbors(knnX.reshape(1, -1)) 

    #use indexing to find hospital ids of neighbors
    benchmarkGroup = []
    for i in list(ind[0][1:]):
        benchmarkGroup.append(featuresDf.iloc[i]['Hospital'].astype(int)) 
        
    return benchmarkGroup