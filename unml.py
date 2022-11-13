# unsupervised machine leaarning quicktest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.preprocessing import StandardScaler

#method
def dimensionality_reduction(data,standardscale='False',method='pca',**kwargs):
    if(standardscale=='True'):
        data=StandardScaler().fit_transform(data)
    '''
    method_map={
        'pca' or 'PCA':PCA(**kwargs).fit_transform(data),
        'tsne' or 'TSNE' or 't-SNE':TSNE(**kwargs).fit_transform(data),
        'umap' or 'UMAP':UMAP(**kwargs).fit_transform(data),
        }
    '''
    if(method=='pca'or'PCA'):
        dr=PCA(**kwargs).fit_transform(data)
    elif(method=='tsne' or 'TSNE' or 't-SNE'):
        dr=TSNE(**kwargs).fit_transform(data)
    elif(method=='umap' or 'UMAP'):
        dr=UMAP(**kwargs).fit_transform(data)
    #dr=method_map[method]()
    return dr

#labels sort and classify
def ML_label(labels,observe,threshold=0.1,test=False):
    maxlb=np.max(labels)
    num=pd.DataFrame(columns=np.arange(0,maxlb+1),index=['rp_num','sum_num','rp_ratio'])
    for i in range(0,maxlb+1):
        n=np.sum((labels==i))
        num.loc['sum_num',i]=n
        r=np.sum((labels==i)&(observe=='repeater'))
        num.loc['rp_num',i]=r
    num.loc['rp_ratio',:]=num.loc['rp_num',:]/num.loc['sum_num',:]
    cf=num.sort_values(by='rp_ratio',axis=1)
    
    new=np.arange(0,maxlb+1)
    old=cf.columns
    new_labels=labels
    for i in range(len(labels)):
        if (labels[i] in new):
            new_labels[i]=new[labels[i]==old]
    cf.columns=new
    rp_lb=new[cf.loc['rp_ratio',:]>=threshold]
    nrp_lb=new[cf.loc['rp_ratio',:]==0]
    oth_lb=new[(cf.loc['rp_ratio',:]<threshold)&(cf.loc['rp_ratio',:]>0)]
    if(test==True):print(cf.loc['rp_ratio',:])
    return new_labels,nrp_lb,oth_lb,rp_lb

#score for manifold methods
def get_mi_score(data,label,embedding,labels_list,test=False,axis=None):
    col=[
    'peak_freq','log_bc_width','log_flux','log_fluence','redshift','fre_width','log_in_duration','log_energy', 'log_luminosity','log_T_B'
]

    cl_data=data.copy()#pd.DataFrame(StandardScaler().fit_transform(data.copy().values),columns=col)
    cl_data['labels']=labels_list
    cl_data[['x','y']]=embedding
    
    if(label==-1):cl_target=cl_data.copy()
    else:cl_target=cl_data[(cl_data['labels']==label)].copy()
    cl_target.drop(columns='labels',inplace=True)
    
    X=cl_target.drop(columns=['x','y'])
    
    y=cl_target[['x','y']]
    if(test==True):
        print('x=')
        print(X)
        print('y=')
        print(y)
    
    if(axis==0):
        cl_MI = pd.DataFrame(
            {
                'Variables': X.columns,
                'x': mutual_info_regression(X, y['x']),
                'y': mutual_info_regression(X, y['y'])
            })
    else:
        cl_MI=pd.DataFrame(
            {
                'Variables': X.columns,
                axis: mutual_info_regression(X, y[axis])
            })
    
    return cl_MI

#plot
def ML_plot2D(origin_data,data_plane,p=12,fontsize=18,title='Machine Learning plot',xlabel=None,ylabel=None,save=None,hue=None,legend=None,color=None):
    sns.set(context='notebook', style='white', rc={'figure.figsize':(p,0.8*p)})
    
    plane=pd.DataFrame(data = data_plane, columns = ['x','y'])
    plane[hue]=origin_data[hue]
    #color=['royalblue','tomato']
    if(color==None):color='pastel'

    if(hue==None):sns.scatterplot(x='x',y='y',data=plane,linewidth=0,palette=color)
    else:sns.scatterplot(x='x',y='y',hue=hue,data=plane,linewidth=0,palette=color)
    
    if(legend!=None):plt.legend(legend)
    plt.title(title,fontsize=fontsize)
    plt.xlabel(xlabel=xlabel,fontsize=fontsize)
    plt.ylabel(ylabel=ylabel,fontsize=fontsize)
    if(save!=None):plt.savefig(save,bbox_inches='tight',dpi=100,pad_inches=0.5)
    return None

def distance(embedding,point):
    embedding=embedding-point
    dis=np.linalg.norm(embedding,axis=1)
    return dis

#classify noise points based on Euclidean distance
""" def noise_points_cl(embedding,labels_list,noise_label=-1):
    for i in range(len(labels_list)):
        if(labels_list[i]==noise_label):
            dis=distance(embedding=embedding,point=embedding[i])
            dis[np.where(labels_list==-1)]=np.max(dis)+1
            labels_list[i]=labels_list[np.where(dis==np.min(dis))]
    return labels_list """