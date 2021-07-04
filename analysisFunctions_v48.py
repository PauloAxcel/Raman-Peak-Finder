import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Gets rid of some warnings in output text
import warnings
warnings.filterwarnings("ignore")
import scipy
from scipy.linalg import solveh_banded
from scipy import signal
from sklearn.decomposition import PCA
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from scipy import stats
import nestle 
import sys
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns			
import random		
from random import randint
from sklearn.metrics.pairwise import euclidean_distances
#import statsmodels.stats.api as sms     # t-test
import statistics as stat               # F-test
from scipy import stats                 # KS Test
from scipy.stats import f               # F-test
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
import scipy
from scipy.linalg import solveh_banded
from scipy import signal
import matplotlib.pyplot as plt
#%matplotlib qt5
import matplotlib.pyplot as plt 

from minisom import MiniSom   
from sklearn import preprocessing
import itertools
from scipy.interpolate import make_interp_spline, BSpline

from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from itertools import groupby

import itertools
import circlify as circ
import math
import math
from scipy import ndimage
import matplotlib.patches as mpatches  
import matplotlib.gridspec as gridspec
from itertools import chain
from sklearn.preprocessing import minmax_scale, scale

font = {'family' : 'Arial',
        'size'   : 22}

plt.rc('font', **font)

# From MatrixExp
def matrix_exp_eigen(U, s, t, x):
    exp_diag = np.diag(np.exp(s * t), 0)
    return U.dot(exp_diag.dot(U.transpose().dot(x))) 

# From LineLaplacianBuilder
def get_line_laplacian_eigen(n):
    assert n > 1
    eigen_vectors = np.zeros([n, n])
    eigen_values = np.zeros([n])

    for j in range(1, n + 1):
        theta = np.pi * (j - 1) / (2 * n)
        sin = np.sin(theta)
        eigen_values[j - 1] = 4 * sin * sin
        if j == 0:
            sqrt = 1 / np.sqrt(n)
            for i in range(1, n + 1):
                eigen_vectors[i - 1, j - 1] = sqrt
        else:
            for i in range(1, n + 1):
                theta = (np.pi * (i - 0.5) * (j - 1)) / n
                math_sqrt = np.sqrt(2.0 / n)
                eigen_vectors[i - 1, j - 1] = math_sqrt * np.cos(theta)
    return eigen_vectors, eigen_values

def smooth2(t, signal):
    dim = signal.shape[0]
    U, s = get_line_laplacian_eigen(dim)
    return matrix_exp_eigen(U, -s, t, signal)


def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e6,
                 max_iters=10, conv_thresh=1e-5, verbose=False):
  '''Computes the asymmetric least squares baseline.
  * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
  smoothness_param: Relative importance of smoothness of the predicted response.
  asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                       Setting p=1 is effectively a hinge loss.
  '''
  smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
  # Rename p for concision.
  p = asymmetry_param
  # Initialize weights.
  w = np.ones(intensities.shape[0])
  for i in range(max_iters):
    z = smoother.smooth(w)
    mask = intensities > z
    new_w = p*mask + (1-p)*(~mask)
    conv = np.linalg.norm(new_w - w)
    if verbose:
      print (i+1, conv)
    if conv < conv_thresh:
      break
    w = new_w
  else:
    print ('ALS did not converge in %d iterations' % max_iters)
  return z


class WhittakerSmoother(object):
  def __init__(self, signal, smoothness_param, deriv_order=1):
    self.y = signal
    assert deriv_order > 0, 'deriv_order must be an int > 0'
    # Compute the fixed derivative of identity (D).
    d = np.zeros(deriv_order*2 + 1, dtype=int)
    d[deriv_order] = 1
    d = np.diff(d, n=deriv_order)
    n = self.y.shape[0]
    k = len(d)
    s = float(smoothness_param)

    # Here be dragons: essentially we're faking a big banded matrix D,
    # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
    diag_sums = np.vstack([
        np.pad(s*np.cumsum(d[-i:]*d[:i]), ((k-i,0),), 'constant')
        for i in range(1, k+1)])
    upper_bands = np.tile(diag_sums[:,-1:], n)
    upper_bands[:,:k] = diag_sums
    for i,ds in enumerate(diag_sums):
      upper_bands[i,-i-1:] = ds[::-1][:i+1]
    self.upper_bands = upper_bands

  def smooth(self, w):
    foo = self.upper_bands.copy()
    foo[-1] += w  # last row is the diagonal
    return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)




def ImportData(inputFilePath):
#    print(inputFilePath)
    
    df = pd.read_csv(inputFilePath,   
                          sep = '\s+',
                          header = None)
    if df.shape[1] == 2:
    
        df = pd.read_csv(inputFilePath, 
                              sep = '\s+',
                              header = None, 
                              skiprows = 1,
                              names = ['Wavenumber', 'Intensity'])
        n_points = sum((df.iloc[0,1]==df.iloc[:,1][df.iloc[0,0]==df.iloc[:,0]]))
        n_raman = df.shape[0]//n_points
        wavenumber = df['Wavenumber']
            
        dataset = []
        label = inputFilePath.split('/')[-1][:11] 
        intensity = df['Intensity']
        flag = 0
        
    else:        
        
        df = pd.read_csv(inputFilePath, 
                              sep = '\s+',
                              header = None, 
                              skiprows = 1,
                              names = ['X','Y','Wavenumber', 'Intensity'])
        n_points = sum((df.iloc[0,1]==df.iloc[:,1][df.iloc[0,0]==df.iloc[:,0]]))
        n_raman = df.shape[0]//n_points
        wavenumber = df['Wavenumber'][0:n_points]
        
        
        dataset = []
        label = inputFilePath.split('/')[-1][:11]
        
#        works for testing
#        label = inputFilePath.split('\\')[-1][:11]

            
        for i in range(df.shape[0]//n_points):
            ind_df = df['Intensity'][i*n_points:n_points*(i+1)].reset_index(drop=True)
            ind_df = ind_df.rename('spectrum '+str(i+1))
            dataset.append(ind_df)
    
        intensity = pd.concat(dataset,axis=1)
#        print(intensity.shape)
        
#        if zscore == 0:
#            pass
#        else:
#            out = np.abs(stats.zscore(intensity.T))
#            
#            if inv == 0:
#                intensity = intensity.T[(out < zscore).all(axis=1)].T
##                print(intensity.shape)
#            else:
#                intensity = intensity.T[(out > zscore).all(axis=1)].T       
##                print(intensity.shape)
        
        if intensity.shape[1] == 1:
            flag = 0
        else:
            flag = 1
        
    return (wavenumber, intensity, n_points, n_raman, label,flag)


#x=wavenumber
#y = spectra
#y2 = pd.DataFrame()
#fig = plt.figure()


def StackPlot(x,y,y2,label,fig): 
#    print(label)
    NUM_COLORS = y.shape[1]
    
    cmaps = plt.get_cmap('gist_rainbow')
    
    colours = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    
    if y2.empty:
        for i in range(y.shape[1]): 
            plt.plot(x, y.iloc[:,i] ,label=label[i],lw=2,color=colours[i])    
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel(' Intensity (a.u.)')
        plt.legend(loc='best', prop={'size':12},frameon=False)
        fig.show()
    else:
        for n in range(y.shape[1]):
            plt.plot(x, y.iloc[:,n] ,label=label[n],lw=2,color=colours[n])
            plt.fill_between(x, y.iloc[:,n] - y2.iloc[:,n] ,y.iloc[:,n] + y2.iloc[:,n] , color='orange', alpha=0.5)
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel(' Intensity (a.u.)')
        plt.legend(loc='best', prop={'size':12},frameon=False)
        fig.show()

#still to fix!
      

        
def ThreeDPlot(x,y,y2,label,ax):
    
    NUM_COLORS = y.shape[1]
    
    cmaps = plt.get_cmap('gist_rainbow')
    
    colours = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    
    if y2.empty:
        x = x.values
        spec = y
        n_max = y.shape[1]
        for n in range(n_max):
            z = spec.iloc[:,n]
            y = np.array([n]*z.shape[0])
            
            ax.plot(x,  [n]* z.shape[0],  z, c='k', lw=4,zorder=abs(n-n_max))
            ax.plot(x,  [n]* z.shape[0],  z, c = colours[n], lw=1,zorder=abs(n-n_max))
            ax.view_init(30,-45)
            verts = [(x[i],y[i], z[i]) for i in range(z.shape[0])]+[(x.min(), y.min(),ax.axes.get_zlim()[0]),(x.max(), y.max(),ax.axes.get_zlim()[0])]
            ax.add_collection3d(Poly3DCollection([verts], alpha=0.2,edgecolor=None))
            
            ax.set_xlabel('\n Raman shift (cm$^{-1}$)',linespacing=3.2)
            ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
            
            ax.set_ylabel('\n\n Sample',linespacing=3.2)
            ax.set_yticks(np.arange(spec.shape[1]))
            labels = [item.get_text() for item in ax.get_yticklabels()]
            labels[n] = label[n]   
            ax.set_yticklabels(labels)
            ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
            
            ax.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax.set_zlabel('Intensity (a.u.)',rotation=90)
            ax.set_zticks([])
            ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    else:
        x = x.values
        spec = y
        spec_std = y2
        n_max = y.shape[1]
        for n in range(n_max):
            z = spec.iloc[:,n]
            zstd = spec_std.iloc[:,n]
            y = np.array([n]*z.shape[0])

            ax.plot(x,  [n]* z.shape[0],  z, c='k', lw=4,zorder=abs(n-n_max))
            ax.plot(x,  [n]* z.shape[0],  z, c=colours[n],lw=1,zorder=abs(n-n_max))
            ax.view_init(30,-45)
       
            verts2 = [(x[i],y[i], z[i]+zstd[i]) for i in range(z.shape[0])]
            for i in range(z.shape[0]):
                verts2.append((x[-(i+1)],y[-(i+1)], z.iloc[-(i+1)]-zstd.iloc[-(i+1)]))

            ax.add_collection3d(Poly3DCollection([verts2], alpha=0.5,color='orange',edgecolor=None))
             
            
            ax.set_xlabel('\n Raman shift (cm$^{-1}$)',linespacing=3.2)
            ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
            
            ax.set_ylabel('\n\n Sample',linespacing=3.2)
            ax.set_yticks(np.arange(spec.shape[1]))
            labels = [item.get_text() for item in ax.get_yticklabels()]
            labels[n] = label[n]   
            ax.set_yticklabels(labels)
            ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
            
            ax.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax.set_zlabel('Intensity (a.u.)',rotation=90)
            ax.set_zticks([])
            ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))


def IndividualPlot(x,y,label):
    if pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    plt.figure(figsize=(9,9/1.618))
    plt.plot(x, y_avg,color='k',label=label,lw=2)    
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel(' Intensity (a.u.)')
    plt.legend(loc='best', prop={'size':12},frameon=False)


    
    
def ShadeStackPlot(x,y1,y2,label,n,fig):
    if y2.empty:
        exit
    else:      
        plt.plot(x, y1+y1.min()*n ,label=label,lw=2)    
        plt.fill_between(x, y1- y2 +y1.min()*n, y1 + y2 +y1.min()*n, color='orange', alpha=0.5)
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel(' Intensity (a.u.)')
        plt.legend(loc='best', prop={'size':12},frameon=False)
        fig.show()
 


#issues with the dataframe representation of 2dim lists

#solves wavenumber differences
def wavdif(wavenumber,spec,spec_std):
    if not all(x == [len(a) for a in wavenumber][0] for x in [len(a) for a in wavenumber]):
        min_val = []
        max_val = []
        for wav in wavenumber:
            min_val.append(min(wav))
            max_val.append(max(wav))
        min_val = max(min_val)
        max_val = min(max_val)
    
        xnew = np.linspace(min_val, max_val, max([len(a) for a in wavenumber]))
        
        spec2 = []
        for wav,spe in zip(wavenumber,spec):
            new_spec = []
            for i in range(spe.shape[1]):
                new_spec.append(make_interp_spline(sorted(wav), spe.iloc[:,i][::-1].values, k=3)(xnew)[::-1])
            spec2.append(pd.DataFrame(np.transpose(new_spec),columns=spe.columns))
        
        spec = spec2
        spec_std = spec2
        wavenumber = pd.Series(xnew)
    
    else:
        wavenumber = wavenumber[0]

    return spec,spec_std,wavenumber


#inputFilePath = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
#                 r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\milk_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
#                 r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\01st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
#                 ]


def spec(inputFilePath):
    
    spec = []
    spec_std = []
    label = []
    wavenumber = []
    for file in inputFilePath:
        spec.append(ImportData(file)[1])
#        if ImportData(file,zscore,inv)[1].std(axis=1).isnull().all():
#            spec_std = pd.DataFrame()
#        else:           
        spec_std.append(ImportData(file)[1])
        
#        works for testing
#        label.append(file.split('\\')[-1][:11])
        
        label.append(file.split('/')[-1][:15]) 
        wavenumber.append(ImportData(file)[0])
        
    #code if the case that slight different wavenumbers are used
    
    spec,spec_std,wavenumber = wavdif(wavenumber,spec,spec_std)

    return (wavenumber, spec , label,spec_std)



def diff(spectra,spectra_std):
    
    new_spec = []
    for i in range(spectra.shape[1]):
         
        if spectra_std.empty:     
            if i == 0:
                minim = spectra.iloc[:,i].min()
                z = spectra.iloc[:,i]-minim
                new_spec.append(z)
                minim = z
            else:
                minim = (spectra.iloc[:,i]-minim).min()
                z = spectra.iloc[:,i]-minim
                new_spec.append(z)
                minim = z 
    
        else:
            
            zero = pd.DataFrame([0]*spectra_std.shape[0])
            spectra_std = pd.concat([spectra_std,zero],axis=1)


            if i == 0:
                minim = spectra.iloc[:,i].min()
                z = spectra.iloc[:,i]-minim+spectra_std.iloc[:,i].max()
                new_spec.append(z)
                minim = z

            else:
                minim = (spectra.iloc[:,i]-minim).min()
                z = spectra.iloc[:,i]-minim+spectra_std.iloc[:,i].max()+abs((spectra_std.iloc[:,i-1]+spectra_std.iloc[:,i])).min()
                new_spec.append(z)
                minim = z   

                
    return pd.DataFrame(new_spec).T

    
    
def ShadePlot(x,y1,y2,label):
    plt.figure(figsize=(9,9/1.618))
    plt.plot(x, y1,color='k',label=label,lw=2)
    plt.fill_between(x, y1- y2 , y1 + y2 , color='orange', alpha=0.5)

    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel(' Intensity (a.u.)')
    plt.legend(loc='best', prop={'size':12},frameon=False)
    
    plt.show()


def round2(n, k):
    return n - n % k

from scipy.interpolate import make_interp_spline, BSpline
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter

def PlotHeatMap(x,y,label,flag):
    xnew = np.linspace(int(x.min()), int(x.max()),  int(round(x.max()-x.min())+1))
    ynew = []
    
    for n in range(y.shape[1]):
        spl = make_interp_spline(sorted(x),y.iloc[:,n][::-1].reset_index(drop=True) ,k=3)
        #add a smoothing filter
        ynew.append(savgol_filter(spl(xnew),11,3))
    ynew = pd.DataFrame(ynew).T    
    ynew =  pd.DataFrame(gaussian_filter(ynew, sigma=3))
    ynew.columns = y.columns
    ynew.index = xnew.astype(int)
    
    tick_step = 50
    tick_min = int(round2(ynew.index.min(), (-1 * tick_step))) # round down
    tick_max = int((round2(ynew.index.max(), (1 * tick_step))) + tick_step)  # round up
    
    xticklabels = range(tick_min, tick_max, tick_step)
    
    xticks = []
    for val in xticklabels:
        idx_pos = ynew.index.get_loc(val)
        xticks.append(idx_pos)
        
    if flag == 1:
        plt.figure(figsize=(9,9/1.618))
        ax = sns.heatmap(ynew.T,xticklabels=xticklabels,vmin=0,vmax=1,cbar=True)
        plt.ylim(-1,y.shape[1]+1)
        plt.xlim(-1,y.shape[0]+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.title(label)
    
    else:
        plt.figure(figsize=(9,9/1.618))
        ax = sns.heatmap(ynew.T,xticklabels=xticklabels,vmin=0,cbar=True)
        plt.ylim(-1,y.shape[1]+1)
        plt.xlim(-1,y.shape[0]+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.title(label)
    

    

def STD(y):
#    if pd.DataFrame(y).shape[1]==1:
#        y_avg = y
#    else:
#        y_avg = y.mean(axis=1)
    y_avg = y.mean(axis=1)
    y_std = y.std(axis=1)    
    return y_avg, y_std
                 
def Baseline(y,asymmetry_param, smoothness_param, max_iters, conv_thresh):
    if pd.DataFrame(y).shape[1]==1 and isinstance(y, pd.DataFrame):
        y_avg = y.iloc[:,0]
    elif pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    return y_avg - als_baseline(y_avg ,asymmetry_param, smoothness_param, max_iters, conv_thresh)


def BaselineNAVG(y,asymmetry_param, smoothness_param, max_iters, conv_thresh):
#    y = pd.DataFrame(y)
    y = y.dropna(axis='columns')
    a = []
    for i in range(y.shape[1]):
        a.append(y.iloc[:,i].values-als_baseline(y.iloc[:,i],asymmetry_param, smoothness_param, max_iters, conv_thresh))    
    z = pd.DataFrame(np.transpose(a))
    z.columns = y.columns
    
    return z    
#ax = plt.axes(projection = '3d')      

#y = spec

def separate_stack(y):
    a = []
    b = []
    
    for j in range(len(y)):
        z = y[j]
        z = z.dropna(axis='columns')
        for i in range(z.shape[1]):
            a.append(z.iloc[:,i])    
            b.append(i)
    sp = pd.DataFrame(np.transpose(a),columns=np.transpose(b))
    
    return sp

def BaselineNAVGS(y,asymmetry_param, smoothness_param, max_iters, conv_thresh):
    a = []
    b = []
    
    for j in range(len(y)):
        z = y[j]
        z = z.dropna(axis='columns')
        for i in range(z.shape[1]):
            a.append(z.iloc[:,i].values-als_baseline(z.iloc[:,i],asymmetry_param, smoothness_param, max_iters, conv_thresh))    
            b.append(i)
    sp = pd.DataFrame(np.transpose(a),columns=np.transpose(b))
    
    return sp


#NOTE THE i != sp.shape[1]-1 GAVE ERROR WHEN THERE ARE ONLY 2 SPECTRA PER MAP... IDK WHY
# COULD POTENTIALLY BE REMOVED?
 
def STDS(sp):
    a = []
    b = []
    c = []
    j = -1
    for i in range(sp.shape[1]):
        if sp.columns[i] > j and i != sp.shape[1]-1: 

            a.append(sp.iloc[:,i])
            j = j + 1

        else:
            j = -1
            y = pd.DataFrame(a)
            a = []

            b.append([y.mean(axis=0)])
            c.append([y.std(axis=0)])
                
    db = pd.DataFrame()
    dc = pd.DataFrame()            

    for bs,cs in zip(b,c):
        db = pd.concat([db,pd.DataFrame(bs).T],axis=1)
        dc = pd.concat([dc,pd.DataFrame(cs).T],axis=1)
        
    db.columns=list(range(db.shape[1]))
    dc.columns=list(range(dc.shape[1]))
           
    return db, dc

def NormalizeNAVGS(sp,zscore,inv,asymmetry_param, smoothness_param, max_iters, conv_thresh):
    a = []
    b = []
    c = []
    j = -1
    for i in range(sp.shape[1]):
        if sp.columns[i] > j and i != sp.shape[1]-1:

            a.append(sp.iloc[:,i])
            j = j + 1

        else:
            j = -1
            y = pd.DataFrame(a)
            a = []
            
            y = NormalizeNAVG(y.T,zscore,inv)

            b.append([y.mean(axis=1)])
            c.append([Baseline(y.std(axis=1),asymmetry_param, smoothness_param, max_iters, conv_thresh)])
                
    db = pd.DataFrame()
    dc = pd.DataFrame()            

    for bs,cs in zip(b,c):
        db = pd.concat([db,pd.DataFrame(bs).T],axis=1)
        dc = pd.concat([dc,pd.DataFrame(cs).T],axis=1)
           
    return db, dc


def Normalize(y,zscore,inv):  
    if zscore == 0:
        pass
    else:
        out = np.abs(stats.zscore(y.T)) 
        
        if inv == 0:
            y = y.T[(out < zscore).all(axis=1)].T
    #                print(intensity.shape)
        else: 
            y = y.T[np.array([not c for c in (out < zscore).all(axis=1)])].T       
    #                print(intensity.shape)
    if pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    return ((y_avg-y_avg.min())/(y_avg.max()-y_avg.min()))
 

def NormalizeNAVG(y,zscore,inv):
    if zscore == 0:
        pass
    else:
        out = np.abs(stats.zscore(y.T))
        
        if inv == 0:
            y = y.T[(out < zscore).all(axis=1)].T
    #                print(intensity.shape)
        else:
            y = y.T[np.array([not c for c in (out < zscore).all(axis=1)])].T             
    #                print(intensity.shape)
    a = []
    for i in range(y.shape[1]):
        a.append((y.iloc[:,i].values-y.iloc[:,i].min())/(y.iloc[:,i].max()-y.iloc[:,i].min()))
    z = pd.DataFrame(np.transpose(a))
    z.columns = y.columns
    
    return z

def NormalizeNSTD(y,ystd,zscore,inv):
    if zscore == 0:
        pass
    else:
        out = np.abs(stats.zscore(y.T))
        
        if inv == 0:
            y = y.T[(out < zscore).all(axis=1)].T
            ystd = ystd.T[(out < zscore).all(axis=1)].T
    #                print(intensity.shape)
        else:
            y = y.T[np.array([not c for c in (out < zscore).all(axis=1)])].T  
            ystd = ystd.T[np.array([not c for c in (out < zscore).all(axis=1)])].T
    #                print(intensity.shape)
    a = []
    for i in range(y.shape[1]):
        a.append((ystd.iloc[:,i].values/y.iloc[:,i].max()))
    z = pd.DataFrame(np.transpose(a))
    z.columns = y.columns
    
    return z


   
def sortData(files,asymmetry_param ,smoothness_param ,max_iters ,conv_thresh,zscore,inv):
    f_target = []
    f_dataset = []
    
    wavenumber,spect,label,spec_std = spec(files)
    
    for spe,lab in zip(spect,label):
    
        baseline = BaselineNAVG(spe,asymmetry_param, smoothness_param, max_iters, conv_thresh)
        norm = NormalizeNAVG(baseline,zscore,inv)
        
        where_are_NaNs = np.isnan(norm)
        if sum(where_are_NaNs.values[0])>0:
            flag = 0
            norm[where_are_NaNs] = 0
        
        for i in range(norm.shape[1]):
            f_target.append(lab)
            f_dataset.append(norm.iloc[:,i])
            flag = 1

                     
    return wavenumber,f_dataset, f_target,flag

#windowmin = 0
#windowmax = 0
#labels = f_target
#norm = f_dataset

def OrganizePCAData(norm,labels,wavenumber,windowmin,windowmax):    
    if windowmin == 0 and windowmin == 0:
        df_pca = pd.DataFrame(norm).reset_index(drop=True)
        target = pd.DataFrame(labels,columns=['sample'])
        
        pca = PCA()
        principalComponents = pca.fit_transform(df_pca)
        columns = ['principal component '+str(i+1) for i in range(principalComponents.shape[1])]
        info = ['PC '+str(i+1)+': '+str(round(pca.explained_variance_ratio_[i]*100,2))+' % \n ' for i in range(principalComponents.shape[1])]
        principalDf = pd.DataFrame(data = principalComponents , columns = columns)
        df = pd.concat([principalDf, target], axis = 1)

    else:
        df_pca = pd.DataFrame(norm).reset_index(drop=True)
        target = pd.DataFrame(labels,columns=['sample'])
        
        region = (wavenumber > windowmin) & (wavenumber < windowmax)
        df_pca = df_pca.T[region].T
        wavenumber = wavenumber[region]
        
        pca = PCA()
        principalComponents = pca.fit_transform(df_pca)
        columns = ['principal component '+str(i+1) for i in range(principalComponents.shape[1])]
        
        info = ['PC '+str(i+1)+': '+str(round(pca.explained_variance_ratio_[i]*100,2))+' % \n ' for i in range(principalComponents.shape[1])]
        principalDf = pd.DataFrame(data = principalComponents , columns = columns)
        df = pd.concat([principalDf, target], axis = 1)
    
        
    return df,target,info,pca,wavenumber

#falg takes care of the gaussian mixture or not

def PCAPlot2D(df,target,info,flag):  
    
    position = []
    
    if flag == 0:
    
#        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
        
            
        fig = plt.figure(figsize=(9,9/1.618))
        ax = fig.add_subplot(1,1,1)
        plt.xlabel(str(info[0]))
        plt.ylabel(str(info[1]))
        
#        colours = []
#        
#        for i in range(len(list(np.unique(target)))):
#            colours.append(cycle[i%len(cycle)])
        
        NUM_COLORS = len(list(np.unique(target)))
        
        cmaps = plt.get_cmap('gist_rainbow')
        
        colours = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
            
#        colours = bin_colours
            
        for label,color in zip(list(np.unique(target)),colours):
            indicesToKeep = df['sample'] == label
    
            new_x = df.loc[indicesToKeep, 'principal component 1']
            new_y = df.loc[indicesToKeep, 'principal component 2']
    
            ax.scatter(new_x, new_y, s =100,alpha=1,label=label,marker='x',color=color)
    
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(loc='best',frameon=False)
            
            position.append([[(new_x.max()-new_x.min())/2,(new_y.max()-new_y.min())/2],
                       (new_x.max()-new_x.min()),
                       (new_y.max()-new_y.min()),label])
            
            plt.show()
        
        
    else:
        
#        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
          
        fig = plt.figure(figsize=(9,9/1.618))
        ax = fig.add_subplot(1,1,1)
        plt.xlabel(str(info[0]))
        plt.ylabel(str(info[1]))
        
#        colours = []
#        
#        for i in range(len(list(np.unique(target)))):
#            colours.append(cycle[i%len(cycle)])
            
#        colours = bin_colours
        
        NUM_COLORS = len(list(np.unique(target)))
        
        cmaps = plt.get_cmap('gist_rainbow')
        
        colours = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
            
        for label,color in zip(list(np.unique(target)),colours):
            indicesToKeep = df['sample'] == label
            
            if df.loc[indicesToKeep].shape[0] < 2:
                
                new_x = df.loc[indicesToKeep, 'principal component 1']
                new_y = df.loc[indicesToKeep, 'principal component 2']
                ax.scatter(new_x, new_y, s =100,alpha=1,label=label,marker='x',color = color)
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(loc='best',frameon=False)
                
            else:
                    
              
                gmm = GaussianMixture(n_components=1).fit(pd.concat([df.loc[indicesToKeep, 'principal component 1']
                           , df.loc[indicesToKeep, 'principal component 2']],axis=1).values)
            
            
               
                for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
#                    print(pos)
#                    print(covar)
                    
                    
                    if covar.shape == (2, 2):
                        U, s, Vt = np.linalg.svd(covar)
                        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
                        width, height = 2 * np.sqrt(s)
                    else:
                        angle = 0
                        width, height = 2 * np.sqrt(covar)
                        
                #draw the 2sigma region
                    ax.add_patch(Ellipse(pos,2*width,2*height,angle,alpha=0.3,color = color))
            
                    
                    new_x = df.loc[indicesToKeep, 'principal component 1']
                    new_y = df.loc[indicesToKeep, 'principal component 2']
                    
                #    ax.scatter(new_x, new_y, c = color , s =100,alpha=1,label=target,marker='x')
                    position.append([pos,width,height,label])
                    
                    range1s = (((new_x < pos[0]+np.sqrt(width/2)) & (new_x > pos[0]-np.sqrt(width/2))) | 
                            ((new_y < pos[1]+np.sqrt(height/2)) & (new_y > pos[1]-np.sqrt(height/2))))
                    ax.scatter(new_x[range1s], new_y[range1s], s =100,alpha=1,label=label,marker='x',color = color)
                
                #    plot_gmm(gmm)
                
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = OrderedDict(zip(labels, handles))
                    plt.legend(loc='best',frameon=False)
                
                plt.show()
    
    return position 


def kdcalc(position,stylestring,dim):
    if dim == 2:
            
        if stylestring == 'binary':
            param = 2
            dist = []

            for i in range(len(position)//param):
                x1 = position[i*param][0][0]
                x2 = position[i*param+1][0][0]
                y1 = position[i*param][0][1]
                y2 = position[i*param+1][0][1]
                r = np.sqrt((x1-x2)**2+(y1-y2)**2) 
                
                w1 =  position[i*param][1]
                w2 = position[i*param+1][1]
                h1 = position[i*param][2]
                h2 = position[i*param+1][2]
                er = abs(np.sqrt((w1/2)**2+(h1/2)**2)-np.sqrt((w2/2)**2+(h2/2)**2))
#                er = np.sqrt(((x1-x2)*w1)**2+((y1-y2)*h1)**2+((x2-x1)*w2)**2+((y2-y1)*h2)**2)/r
                
                dist.append([r , er , position[i*param][-1]+' '+position[i*param+1][-1]])
                           
                
        elif stylestring == 'trinary' or stylestring == 'inter':
            param = 3
            dist0 = []
            dist1 = []
            dist2 = []
            for i in range(len(position)//param):
                x1 = position[i*param][0][0]
                x2 = position[i*param+1][0][0]
                x3 = position[i*param+2][0][0]
                y1 = position[i*param][0][1]
                y2 = position[i*param+1][0][1]
                y3 = position[i*param+2][0][1]
                
                r1 = np.sqrt((x1-x2)**2+(y1-y2)**2)
                r2 = np.sqrt((x1-x3)**2+(y1-y3)**2)
                r3 = np.sqrt((x2-x3)**2+(y2-y3)**2)
                
                w1 = position[i*param][1]
                w2 = position[i*param+1][1]
                w3 = position[i*param+2][1]
                h1 = position[i*param][2]
                h2 = position[i*param+1][2]
                h3 = position[i*param+2][2]
                
                
                er1 = abs(np.sqrt((w1/2)**2+(h1/2)**2)-np.sqrt((w2/2)**2+(h2/2)**2))
                er2 = abs(np.sqrt((w1/2)**2+(h1/2)**2)-np.sqrt((w3/2)**2+(h3/2)**2))
                er3 = abs(np.sqrt((w2/2)**2+(h2/2)**2)-np.sqrt((w3/2)**2+(h3/2)**2))
#                er1 = np.sqrt(((x1-x2)*w1)**2+((y1-y2)*h1)**2+((x2-x1)*w2)**2+((y2-y1)*h2)**2)/r1
#                er2 = np.sqrt(((x1-x3)*w1)**2+((y1-y3)*h1)**2+((x3-x1)*w3)**2+((y3-y1)*h3)**2)/r2
#                er3 = np.sqrt(((x2-x3)*w2)**2+((y2-y3)*h2)**2+((x3-x2)*w3)**2+((y3-y2)*h3)**2)/r3

                
                dist0.append([r1 , er1 , position[i*param][-1]+' '+position[i*param+1][-1]])
        
                dist1.append([r2 , er2 , position[i*param][-1]+' '+position[i*param+2][-1]])
                
                dist2.append([r3 , er3 , position[i*param+1][-1]+' '+position[i*param+2][-1]])
            dist = []
            for d in dist0+dist1+dist2:
                dist.append(d)
    
        else:
            param = 1
        
            dist = []
            for i in range(len(position)//param):
                for j in range(len(position)//param):
                    if j>i:
                        x1 = position[i*param][0][0]
                        x2 = position[j*param][0][0]

                        y1 = position[i*param][0][1]
                        y2 = position[j*param][0][1]
   
                        r = np.sqrt((x1-x2)**2+(y1-y2)**2)
                        
                        w1 = position[i*param][1]
                        w2 = position[j*param][1]
                        h1 = position[i*param][2]
                        h2 = position[j*param][2]
                        
                        er = abs(np.sqrt((w1/2)**2+(h1/2)**2)-np.sqrt((w2/2)**2+(h2/2)**2))
#                        er = np.sqrt(((x1-x2)*w1)**2+((y1-y2)*h1)**2+((x2-x1)*w2)**2+((y2-y1)*h2)**2)/r
                
                        dist.append([r, er,position[i*param][-1]+' '+position[i*param+1][-1]])

    elif dim == 3:
            
        if stylestring == 'binary':
            param = 2
            dist = []
            for i in range(len(position)//param):
                x1 = position[i*param][0][0]
                x2 = position[i*param+1][0][0]
                y1 = position[i*param][0][1]
                y2 = position[i*param+1][0][1]
                z1 = position[i*param][0][2]
                z2 = position[i*param+1][0][2]
                
                r = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
                
                w1 = position[i*param][1]
                w2 = position[i*param+1][1]
                h1 = position[i*param][2]
                h2 = position[i*param+1][2]
                zed1 = position[i*param][3]
                zed2 = position[i*param+1][3]
                
                er = abs(np.sqrt((w1/2)**2+(h1/2)**2+(zed1/2)**2)-np.sqrt((w2/2)**2+(h2/2)**2+(zed2/2)**2))
#                er = np.sqrt(((x1-x2)*w1)**2+((y1-y2)*h1)**2+((z1-z2)*zed1)**2+((x2-x1)*w2)**2+((y2-y1)*h2)**2+((z2-z1)*zed2)**2)/r
                
                dist.append([r, er, position[i*param][-1]+' '+position[i*param+1][-1]])
                           
                
        elif stylestring == 'trinary' or stylestring == 'inter':
            param = 3
            dist0 = []
            dist1 = []
            dist2 = []
            for i in range(len(position)//param):
                
                x1 = position[i*param][0][0]
                x2 = position[i*param+1][0][0]
                x3 = position[i*param+2][0][0]
                y1 = position[i*param][0][1]
                y2 = position[i*param+1][0][1]
                y3 = position[i*param+2][0][1]
                z1 = position[i*param][0][2]
                z2 = position[i*param+1][0][2]
                z3 = position[i*param+2][0][2]
                
                r1 = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
                r2 = np.sqrt((x1-x3)**2+(y1-y3)**2+(z1-z3)**2)
                r3 = np.sqrt((x2-x3)**2+(y2-y3)**2+(z2-z3)**2)
                
                w1 = position[i*param][1]
                w2 = position[i*param+1][1]
                w3 = position[i*param+2][1]
                h1 = position[i*param][2]
                h2 = position[i*param+1][2]
                h3 = position[i*param+2][2]
                zed1 = position[i*param][3]
                zed2 = position[i*param+1][3]
                zed3 = position[i*param+2][3]
                
                er1 = abs(np.sqrt((w1/2)**2+(h1/2)**2+(zed1/2)**2)-np.sqrt((w2/2)**2+(h2/2)**2+(zed2/2)**2))
                er2 = abs(np.sqrt((w1/2)**2+(h1/2)**2+(zed1/2)**2)-np.sqrt((w3/2)**2+(h3/2)**2+(zed3/2)**2))
                er3 = abs(np.sqrt((w3/2)**2+(h3/2)**2+(zed3/2)**2)-np.sqrt((w2/2)**2+(h2/2)**2+(zed2/2)**2))
                
#                er1 = np.sqrt(((x1-x2)*w1)**2+((y1-y2)*h1)**2+((z1-z2)*zed1)**2+((x2-x1)*w2)**2+((y2-y1)*h2)**2+((z2-z1)*zed2)**2)/r1
#                er2 = np.sqrt(((x1-x3)*w1)**2+((y1-y3)*h1)**2+((z1-z3)*zed1)**2+((x3-x1)*w3)**2+((y3-y1)*h3)**2+((z3-z1)*zed3)**2)/r2
#                er3 = np.sqrt(((x2-x3)*w2)**2+((y2-y3)*h2)**2+((z2-z3)*zed2)**2+((x3-x2)*w3)**2+((y3-y2)*h3)**2+((z3-z2)*zed3)**2)/r3

                
                dist0.append([r1 , er1 , position[i*param][-1]+' '+position[i*param+1][-1]])
        
                dist1.append([r2 , er2 , position[i*param][-1]+' '+position[i*param+2][-1]])
                
                dist2.append([r3 , er3 , position[i*param+1][-1]+' '+position[i*param+2][-1]])
            dist = []
            for d in dist0+dist1+dist2:
                dist.append(d)
    
        else:
            param = 1
        
            dist = []
            for i in range(len(position)//param):
                for j in range(len(position)//param):
                    if j>i:
                        x1 = position[i*param][0][0]
                        x2 = position[j*param][0][0]

                        y1 = position[i*param][0][1]
                        y2 = position[j*param][0][1]
                        
                        z1 = position[i*param][0][2]
                        z2 = position[j*param][0][2]
   
                        r = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
                        
                        w1 =  position[i*param][1]
                        w2 = position[j*param][1]
                        h1 = position[i*param][2]
                        h2 = position[j*param][2]
                        zed1 = position[i*param][3]
                        zed2 = position[j*param][3]
                        
                        er = abs(np.sqrt((w1/2)**2+(h1/2)**2+(zed1/2)**2)-np.sqrt((w2/2)**2+(h2/2)**2+(zed2/2)**2))
                        
#                        er = np.sqrt(((x1-x2)*w1)**2+((y1-y2)*h1)**2+((z1-z2)*zed1)**2+((x2-x1)*w2)**2+((y2-y1)*h2)**2+((z2-z1)*zed2)**2)/r
                        
                        dist.append([r, er, position[i*param][-1]+' '+position[i*param+1][-1]])
                    
        
    return [str(round(e[0],2))+' +/- '+str(round(e[1],2))+' '+e[2]+'\n' for e in dist]
                
                

from scipy import stats
from statannot import add_stat_annotation 
import itertools

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

def Encoder(df):
  columnsToEncode = list(df.select_dtypes(include=['category','object']))
  le = LabelEncoder()
  for feature in columnsToEncode:
      try:
          df[feature] = le.fit_transform(df[feature])
      except:
          print('Error encoding '+feature)
  return df


from sklearn.model_selection import train_test_split

from sklearn.neighbors import (KNeighborsClassifier,  NeighborhoodComponentsAnalysis)
from sklearn.metrics import confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from scipy import interp
from itertools import cycle
from sklearn.preprocessing import label_binarize

def plot_roc_curve(finalDf,fig3,axs3,class_names):
    
    sel = []
    for cl in class_names:
        index = cl == finalDf.iloc[:,-1]
        
        z_scores = stats.zscore(finalDf.loc[index].iloc[:,:-1])     
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 5).all(axis=1)
#        new_df = finalDf.loc[index].iloc[:,:-1][filtered_entries]
        
        sel.append(finalDf.loc[index][filtered_entries])
    
    Df = pd.concat([sel[0],sel[1]])
    
    X = Df.iloc[:,:-1].values
    y = Df.iloc[:,-1].values
    
    y_n1 = label_binarize(y,classes= class_names)
    y_n2 = label_binarize(y[::-1],classes= class_names)
    y_n = []
    
    for y1, y2 in zip(y_n1,y_n2):
        y_n.append([y1[0],y2[0]])

    random_state = 42    
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, np.array(y_n), test_size=.2,  random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    lw = 2
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
#    fig,axs3 = plt.subplots(figsize=(9,9/1.618))
    axs3.plot(fpr["micro"], tpr["micro"],
             label='maROCc (A = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    axs3.plot(fpr["macro"], tpr["macro"],
             label='MaROCc (A = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['orange','sienna','mediumpurple'])
    for i, color in zip(range(n_classes), colors):
        axs3.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROCcc {0} (A= {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    axs3.plot([0, 1], [0, 1], 'k--', lw=lw)
    axs3.set_xlim([0.0, 1.0])
    axs3.set_ylim([0.0, 1.05])
    axs3.set_xlabel('FPR')
    axs3.set_ylabel('TPR')
#    axs3.set_title('Some extension of Receiver operating characteristic to multi-class')
    axs3.legend(loc="best",frameon=False,fontsize='x-small')
    plt.tight_layout()




def plot_confusion_matrix(cm, fig , ax , classes, cmap=plt.cm.Blues):  
    
    sumx = cm[0][0] + cm[0][1]
    sumy = cm[1][0] + cm[1][1]
    
    TP = cm[0][0]/sumx
    FN = cm[0][1]/sumx
    FP = cm[1][0]/sumy
    TN = cm[1][1]/sumy
        
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    cm_new = 100*np.array([[TP,FN,TPR],
                       [FP,TN,TNR],
                       [PPV,NPV,ACC]])
    
    classes = classes + ['Measures']
    
    cl = []
    for clas in classes:
        cl.append(clas[:5])

#    fig,ax = plt.subplots()
    ax.imshow(cm_new, interpolation='nearest', cmap=cmap)
    tick_marks = list(np.arange(len(cl)))
    
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(cl,rotation=45)
    ax.set_xlabel('Predicted label')
    ax.set_xlim(0-0.5,2+0.5)

    
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(cl,rotation=45)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_ylim(0-0.5,2+0.5)

    
    lab2 = ['P','NPV','A']
    tick_marks2 = list(np.arange(len(lab2)))
    lab3 = ['Se','Sp','A']
    tick_marks3 = list(np.arange(len(lab3)))
    
    ax2 = ax.twiny().twinx()
    ax2.xaxis.set_label_position('bottom') 
    ax2.set_xticks(tick_marks2)
    ax2.set_xticklabels(lab2,rotation=45)
    ax2.set_yticks(tick_marks3)
    ax2.set_yticklabels(lab3,rotation=45)
    ax2.set_xlim(0-0.5,2+0.5)
    ax2.set_ylim(0-0.5,2+0.5)
    
    r = plt.Rectangle([-0.5,-0.5], 2,2, facecolor="none", edgecolor="k", linewidth=3)
    plt.gca().add_patch(r)
       

    thresh = cm_new.max() / 2.
    for i, j in itertools.product(range(cm_new.shape[0]), range(cm_new.shape[1])):
        ax.text(j, i, str(round(cm_new[i, j],1))+'%',
                 horizontalalignment="center",
                 color="white" if cm_new[i, j] > thresh else "black")

    plt.tight_layout()

import math

def dist_k_calc(finalDf,stylestring,targets,position,colours,dim):
    
#    colours = bin_colours
#    colours = []
#    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
#    for i in range(len(list(np.unique(targets)))):
#        colours.append(cycle[i%len(cycle)])
    
    if stylestring == 'binary':
        
        param = 2
        
        fig, axs = plt.subplots(math.ceil(len(list(np.unique(targets)))//param/3), 
                                len(list(np.unique(targets)))//2 if len(list(np.unique(targets)))//2<3 else 3 ,
                                figsize=(5*len(list(np.unique(targets)))//param,5))
        
        fig2, axs2 = plt.subplots(math.ceil(len(list(np.unique(targets)))//param/3),
                                  len(list(np.unique(targets)))//2 if len(list(np.unique(targets)))//2<3 else 3 ,
                                  figsize=(5*len(list(np.unique(targets)))//param,5))
        
        fig3, axs3 = plt.subplots(math.ceil(len(list(np.unique(targets)))//param/3),
                                  len(list(np.unique(targets)))//2 if len(list(np.unique(targets)))//2<3 else 3 ,
                                  figsize=(5*len(list(np.unique(targets)))//param,5))
        
        if len(list(np.unique(targets)))//2>1:
            axs,axs2,axs3 = axs.ravel(),axs2.ravel(),axs3.ravel()
        
        for n in range(len(list(np.unique(targets)))//param):
              
            indexes = np.unique(targets, return_index=True)[1] 
            indicesToKeep = finalDf['sample'] == np.concatenate([targets.values[index] for index in sorted(indexes)])[n*param]
            df_substrate = finalDf[indicesToKeep].reset_index(drop=True)
            
            indexes2 = np.unique(targets, return_index=True)[1]
            indicesToKeep2 = finalDf['sample'] == np.concatenate([targets.values[index] for index in sorted(indexes2)])[n*param+1]
            df_sugars = finalDf[indicesToKeep2].reset_index(drop=True)
            
            if dim == 2:
                
                PC1 = df_substrate.iloc[:,0]
                PC2 = df_substrate.iloc[:,1]
                
                PC10 = df_sugars.iloc[:,0]
                PC20 = df_sugars.iloc[:,1]
                               
                df0 = pd.concat([PC1,PC2,df_substrate.iloc[:,-1]],axis=1)
                df1 = pd.concat([PC10,PC20,df_sugars.iloc[:,-1]],axis=1)
                df = pd.concat([df0,df1],axis=0).reset_index(drop=True)
                df.columns =  ['altered PC1','altered PC2','sample']
                
                df = Encoder(df)

                cmap_light = ListedColormap(['dimgrey', 'gainsboro'])
                
                n_neighbors = param
                random_state = 42

                
                X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.2, stratify=df.iloc[:,-1], random_state=random_state)
                
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)

                knn.fit(X_train, y_train)

                scores = cross_val_score(knn, df.iloc[:,:-1], df.iloc[:,-1], cv=5,scoring='accuracy')
                
                x_min, x_max = df.iloc[:,0].min() - 1, df.iloc[:,0].max() + 1
                y_min, y_max = df.iloc[:,1].min() - 1, df.iloc[:,1].max() + 1
                h=0.05
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))
                
                Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

                Z = Z.reshape(xx.shape)
                
                if len(list(np.unique(targets)))//2>1:
                    axs[n].contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
                    axs[n].plot(PC1,PC2,'o',label = list(np.unique(targets))[n*param],color = colours[n*param])
                    axs[n].plot(PC10,PC20,'o',label = list(np.unique(targets))[n*param+1],color = colours[n*param+1])
                    axs[n].legend(loc='best',frameon=False,fontsize='x-small')
                    axs[n].text(xx.min()*0.9, yy.min()*0.9, 'Sep : %0.1f %% +/- %0.1f %%' % (scores.mean()*100, scores.std()*100), size=15)
                    
                else:
                    axs.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
                    axs.plot(PC1,PC2,'o',label = list(np.unique(targets))[n*param],color = colours[n*param])
                    axs.plot(PC10,PC20,'o',label = list(np.unique(targets))[n*param+1],color = colours[n*param+1])
                    axs.legend(loc='best',frameon=False,fontsize='x-small')
                    axs.text(xx.min()*0.9, yy.min()*0.9, 'Sep : %0.1f %% +/- %0.1f %%' % (scores.mean()*100, scores.std()*100), size=15)


           
            class_names = [list(np.unique(targets))[n*param],list(np.unique(targets))[n*param+1]]
            y_pred = knn.fit(X_train, y_train).predict(X_test)
            cnf_matrix = confusion_matrix(y_test, y_pred)
            
            if len(list(np.unique(targets)))//2>1:                
                plot_confusion_matrix(cnf_matrix, fig2 , axs2[n] , classes=class_names)          
                plot_roc_curve(finalDf,fig3,axs3[n],class_names) 
            else:
                plot_confusion_matrix(cnf_matrix, fig2 , axs2 , classes=class_names)          
                plot_roc_curve(finalDf,fig3,axs3,class_names) 
            

#stylestring = 'trinary'
#targets = target
#finalDf = df
#position = PCAPlot3D(df,target,info,ax,flag)

def dist_calc(finalDf,stylestring,targets,position,dim):
    if stylestring == 'binary':
        
        param = 2
        dist_substrate = []
        dist_matrix = []
        for n in range(len(list(np.unique(targets)))//param):
            dist_sub = []
            dist_sug = []
            dist_base = []
            
            indexes = np.unique(targets, return_index=True)[1] 
            indicesToKeep = finalDf['sample'] == np.concatenate([targets.values[index] for index in sorted(indexes)])[n*param]
            df_substrate = finalDf[indicesToKeep]


            indexes2 = np.unique(targets, return_index=True)[1]
            indicesToKeep2 = finalDf['sample'] == np.concatenate([targets.values[index] for index in sorted(indexes2)])[n*param+1]
            df_sugars = finalDf[indicesToKeep2]
            
            if dim == 2:
                w1 = position[n*param][1]*2
                h1 = position[n*param][2]*2
                w2 = position[n*param+1][1]*2
                h2 = position[n*param+1][2]*2
            if dim == 3:
                w1 = position[n*param][1]*2
                h1 = position[n*param][2]*2
                z1 = position[n*param][3]*2
                
                w2 = position[n*param+1][1]*2
                h2 = position[n*param+1][2]*2
                z2 = position[n*param+1][3]*2
            
    
            for i in range(df_substrate.shape[0]):
                x = df_substrate.iloc[i,0]
                y = df_substrate.iloc[i,1]
                z = df_substrate.iloc[i,2]
                rho = np.sqrt(x**2+y**2)
                r = np.sqrt(x**2+y**2+z**2)
                th = abs(np.arctan(y/x))
                phi = np.arccos(z/r)
                
                if dim == 2:
                    dist_substrate.append([rho*th , df_substrate.iloc[i,-1]])
                    dist_sub.append([x/w1 , y/h1 , df_substrate.iloc[i,-1]])
                elif dim == 3:
                    dist_substrate.append([r* phi, df_substrate.iloc[i,-1]])
                    dist_sub.append([x/w1 , y/h1 , z/z1, df_substrate.iloc[i,-1]])

                    
            
            for i in range(df_sugars.shape[0]):
                x = df_sugars.iloc[i,0]
                y = df_sugars.iloc[i,1]
                z = df_sugars.iloc[i,2]
                rho = np.sqrt(x**2+y**2)
                r = np.sqrt(x**2+y**2+z**2)
                th = abs(np.arctan(y/x))
                phi = np.arccos(z/r)
                
                if dim == 2:
                    dist_substrate.append([rho*th , df_sugars.iloc[i,-1]])
                    dist_sug.append([x/w2 , y/h2 , df_sugars.iloc[i,-1]])
                if dim == 3:
                    dist_substrate.append([r* phi, df_sugars.iloc[i,-1]])
                    dist_sug.append([x/w2 , y/h2 , z/z2, df_sugars.iloc[i,-1]])
                        
                    
            dist_sug = pd.DataFrame(dist_sug)
            dist_sub = pd.DataFrame(dist_sub)
            
            if dim == 2:
                for i in range(dist_sug.shape[0]):
                    x0 = dist_sug.iloc[i,0]
                    y0 = dist_sug.iloc[i,1]
                    for j in range(dist_sub.shape[0]):
                        x1 = dist_sub.iloc[j,0]
                        y1 = dist_sub.iloc[j,1]
                        r = np.sqrt((x0-x1)**2+(y0-y1)**2)
                        dist_matrix.append([r,dist_sub.iloc[j,-1][:11] + ' - ' + dist_sug.iloc[i,-1][:11]])
                        
            if dim == 3:
                for i in range(dist_sug.shape[0]):
                    x0 = dist_sug.iloc[i,0]
                    y0 = dist_sug.iloc[i,1]
                    z0 = dist_sug.iloc[i,2]
                    for j in range(dist_sub.shape[0]):
                        x1 = dist_sub.iloc[j,0]
                        y1 = dist_sub.iloc[j,1]
                        z1 = dist_sub.iloc[j,2]
                        r = np.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
                        dist_matrix.append([r,dist_sub.iloc[j,-1][:11] + ' - ' + dist_sug.iloc[i,-1][:11]])
                        

    elif stylestring == 'trinary' or stylestring == 'inter':
        
        param = 3
        dist_substrate = []
        dist_matrix = []
        for n in range(len(list(np.unique(targets)))//param):
            dist_sub = []
            dist_sug = []
            dist_oth = []
            dist_base = []
            
            indexes = np.unique(targets, return_index=True)[1]
            indicesToKeep = finalDf['sample'] == np.concatenate([targets.values[index] for index in sorted(indexes)])[n*param]
            df_substrate = finalDf[indicesToKeep]

            indexes2 = np.unique(targets, return_index=True)[1]
            indicesToKeep2 = finalDf['sample'] == np.concatenate([targets.values[index] for index in sorted(indexes2)])[n*param+1]
            df_sugars = finalDf[indicesToKeep2]
                     
            indexes3 = np.unique(targets, return_index=True)[1]
            indicesToKeep3 = finalDf['sample'] == np.concatenate([targets.values[index] for index in sorted(indexes3)])[n*param+2]
            df_other = finalDf[indicesToKeep3]


            if dim == 2:
                w1 = position[n*param][1]*2
                h1 = position[n*param][2]*2
                w2 = position[n*param+1][1]*2
                h2 = position[n*param+1][2]*2
                w3 = position[n*param+2][1]*2
                h3 = position[n*param+2][2]*2
                
            if dim == 3:
                w1 = position[n*param][1]*2
                h1 = position[n*param][2]*2
                z1 = position[n*param][3]*2
                
                w2 = position[n*param+1][1]*2
                h2 = position[n*param+1][2]*2
                z2 = position[n*param+1][3]*2
                
                w3 = position[n*param+2][1]*2
                h3 = position[n*param+2][2]*2
                z3 = position[n*param+2][3]*2
    
            for i in range(df_substrate.shape[0]):
                x = df_substrate.iloc[i,0]
                y = df_substrate.iloc[i,1]
                z = df_substrate.iloc[i,2]
                rho = np.sqrt(x**2+y**2)
                r = np.sqrt(x**2+y**2+z**2)
                th = abs(np.arctan(y/x))
                phi = np.arccos(z/r)
                if dim == 2:
                    dist_substrate.append([rho*th , df_substrate.iloc[i,-1]])
                    dist_sub.append([x/w1 , y/h1 , df_substrate.iloc[i,-1]])
                elif dim == 3:
                    dist_substrate.append([r* phi, df_substrate.iloc[i,-1]])
                    dist_sub.append([x/w1 , y/h1 , z/z1, df_substrate.iloc[i,-1]])
                
            for i in range(df_sugars.shape[0]):
                x = df_sugars.iloc[i,0]
                y = df_sugars.iloc[i,1]
                z = df_sugars.iloc[i,2]
                rho = np.sqrt(x**2+y**2)
                r = np.sqrt(x**2+y**2+z**2)
                th = abs(np.arctan(y/x))
                phi = np.arccos(z/r)
                
                if dim == 2:
                    dist_substrate.append([rho*th , df_sugars.iloc[i,-1]])
                    dist_sug.append([x/w2 , y/h2 , df_sugars.iloc[i,-1]])
                if dim == 3:
                    dist_substrate.append([r* phi, df_sugars.iloc[i,-1]])
                    dist_sug.append([x/w2 , y/h2 , z/z2, df_sugars.iloc[i,-1]])
                    
            for i in range(df_other.shape[0]):
                x = df_other.iloc[i,0]
                y = df_other.iloc[i,1]
                z = df_other.iloc[i,2]
                rho = np.sqrt(x**2+y**2)
                r = np.sqrt(x**2+y**2+z**2)
                th = abs(np.arctan(y/x))
                phi = np.arccos(z/r)
                
                if dim == 2:
                    dist_substrate.append([rho*th , df_other.iloc[i,-1]])
                    dist_oth.append([x/w3 , y/h3 , df_other.iloc[i,-1]])
                if dim == 3:
                    dist_substrate.append([r* phi, df_other.iloc[i,-1]])
                    dist_oth.append([x/w3 , y/h3 , z/z3, df_other.iloc[i,-1]])
                
            dist_sug = pd.DataFrame(dist_sug)
            dist_sub = pd.DataFrame(dist_sub)
            dist_oth = pd.DataFrame(dist_oth)
            
            
            if dim == 2:
                for i in range(dist_sug.shape[0]):
                    x0 = dist_sug.iloc[i,0]
                    y0 = dist_sug.iloc[i,1]
                    for j in range(dist_sub.shape[0]):
                        x1 = dist_sub.iloc[j,0]
                        y1 = dist_sub.iloc[j,1]
                        r = np.sqrt((x0-x1)**2+(y0-y1)**2)
                        dist_matrix.append([r,dist_sub.iloc[j,-1][:11] + ' - ' + dist_sug.iloc[i,-1][:11]])
                        
                for i in range(dist_oth.shape[0]):
                    x0 = dist_oth.iloc[i,0]
                    y0 = dist_oth.iloc[i,1]
                    for j in range(dist_sub.shape[0]):
                        x1 = dist_sub.iloc[j,0]
                        y1 = dist_sub.iloc[j,1]
                        r = np.sqrt((x0-x1)**2+(y0-y1)**2)
                        dist_matrix.append([r,dist_sub.iloc[j,-1][:11] + ' - ' + dist_oth.iloc[i,-1][:11]])
                        
                for i in range(dist_oth.shape[0]):
                    x0 = dist_oth.iloc[i,0]
                    y0 = dist_oth.iloc[i,1]
                    for j in range(dist_sug.shape[0]):
                        x1 = dist_sug.iloc[j,0]
                        y1 = dist_sug.iloc[j,1]
                        r = np.sqrt((x0-x1)**2+(y0-y1)**2)
                        dist_matrix.append([r,dist_sug.iloc[j,-1][:11] + ' - ' + dist_oth.iloc[i,-1][:11]])            
            
            if dim == 3:
                for i in range(dist_sug.shape[0]):
                    x0 = dist_sug.iloc[i,0]
                    y0 = dist_sug.iloc[i,1]
                    z0 = dist_sug.iloc[i,2]
                    for j in range(dist_sub.shape[0]):
                        x1 = dist_sub.iloc[j,0]
                        y1 = dist_sub.iloc[j,1]
                        z1 = dist_sub.iloc[j,2]
                        r = np.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
                        dist_matrix.append([r,dist_sub.iloc[j,-1][:11] + ' - ' + dist_sug.iloc[i,-1][:11]])
                        
                        
                for i in range(dist_oth.shape[0]):
                    x0 = dist_oth.iloc[i,0]
                    y0 = dist_oth.iloc[i,1]
                    z0 = dist_oth.iloc[i,2]
                    for j in range(dist_sub.shape[0]):
                        x1 = dist_sub.iloc[j,0]
                        y1 = dist_sub.iloc[j,1]
                        z1 = dist_sub.iloc[j,2]
                        r = np.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
                        dist_matrix.append([r,dist_sub.iloc[j,-1][:11] + ' - ' + dist_oth.iloc[i,-1][:11]])
                        
                for i in range(dist_oth.shape[0]):
                    x0 = dist_oth.iloc[i,0]
                    y0 = dist_oth.iloc[i,1]
                    z0 = dist_oth.iloc[i,2]
                    for j in range(dist_sug.shape[0]):
                        x1 = dist_sug.iloc[j,0]
                        y1 = dist_sug.iloc[j,1]
                        z1 = dist_sug.iloc[j,2]
                        r = np.sqrt((x0-x1)**2+(y0-y1)**2)
                        dist_matrix.append([r,dist_sug.iloc[j,-1][:11] + ' - ' + dist_oth.iloc[i,-1][:11]])      
            

    else:
        param = 1
        dist_substrate = []
        dist_matrix = []
        dist_base = []
#        dist_sub = []
        for n in range(len(list(np.unique(targets)))//param):
            dist_sub = []
            for k in range(len(list(np.unique(targets)))//param):
                if n == 0 and k == 0:
                    indexes = np.unique(targets, return_index=True)[1]
                    indicesToKeep = finalDf['sample'] == np.concatenate([targets.values[index] for index in sorted(indexes)])[n*param]
                    df_substrate = finalDf[indicesToKeep]
                    
                    if dim == 2:
                        w1 = position[n*param][1]*2
                        h1 = position[n*param][2]*2
                        
                    if dim == 3:
                        w1 = position[n*param][1]*2
                        h1 = position[n*param][2]*2
                        z1 = position[n*param][3]*2
                        
                    
                    for i in range(df_substrate.shape[0]):
                        x = df_substrate.iloc[i,0]
                        y = df_substrate.iloc[i,1]
                        z = df_substrate.iloc[i,2]
                        rho = np.sqrt(x**2+y**2)
                        r = np.sqrt(x**2+y**2+z**2)
                        th = abs(np.arctan(y/x))
                        phi = np.arccos(z/r)
                        
                        if dim == 2:
                            dist_sub.append([x/w1,y/h1,df_substrate.iloc[i,-1]])
                            dist_substrate.append([rho*th, df_substrate.iloc[i,-1]])
                        elif dim == 3:
                            dist_sub([x/w1,y/h1,z/z1,df_substrate.iloc[i,-1]])
                            dist_substrate.append([r*phi, df_substrate.iloc[i,-1]])
                    
                    dist_sub = pd.DataFrame(dist_sub)
                    
                    if dim == 2:
                        
                        for i in range(dist_sub.shape[0]):
                            x0 = dist_sub.iloc[i,0]
                            y0 = dist_sub.iloc[i,1]
                            for j in range(dist_sub.shape[0]):
                                if j>i:
                                    x1 = dist_sub.iloc[j,0]
                                    y1 = dist_sub.iloc[j,1]
                                    r = np.sqrt((x0-x1)**2+(y0-y1)**2)
                                    dist_base.append([r,dist_sub.iloc[j,-1][:11]])
                                    
                    if dim == 3:
                        
                        for i in range(dist_sub.shape[0]):
                            x0 = dist_sub.iloc[i,0]
                            y0 = dist_sub.iloc[i,1]
                            z0 = dist_sub.iloc[i,2]
                            for j in range(dist_sub.shape[0]):
                                if j>i:
                                    x1 = dist_sub.iloc[j,0]
                                    y1 = dist_sub.iloc[j,1]
                                    z1 = dist_sub.iloc[j,2]
                                    r = np.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
                                    dist_base.append([r,dist_sub.iloc[j,-1][:11]])
                                
                elif n >= k:
                    pass
                
                else:
                    dist_sug = []
    
                    indexes = np.unique(targets, return_index=True)[1]
                    indicesToKeep = finalDf['sample'] == np.concatenate([targets.values[index] for index in sorted(indexes)])[n*param]
                    df_substrate = finalDf[indicesToKeep]


                    indexes2 = np.unique(targets, return_index=True)[1]
                    indicesToKeep2 = finalDf['sample'] == np.concatenate([targets.values[index] for index in sorted(indexes2)])[k*param]
                    df_sugars = finalDf[indicesToKeep2]
                    
                    if dim == 2:
                        w1 = position[n*param][1]*2
                        h1 = position[n*param][2]*2
                        w2 = position[n*param+1][1]*2
                        h2 = position[n*param+1][2]*2
                    if dim == 3:
                        w1 = position[n*param][1]*2
                        h1 = position[n*param][2]*2
                        z1 = position[n*param][3]*2
                        
                        w2 = position[n*param+1][1]*2
                        h2 = position[n*param+1][2]*2
                        z2 = position[n*param+1][3]*2
                    

                    for i in range(df_substrate.shape[0]):
                        x = df_substrate.iloc[i,0]
                        y = df_substrate.iloc[i,1]
                        z = df_substrate.iloc[i,2]
                        rho = np.sqrt(x**2+y**2)
                        r = np.sqrt(x**2+y**2+z**2)
                        th = abs(np.arctan(y/x))
                        phi = np.arccos(z/r)
                        
                        if dim == 2:
                            dist_substrate.append([rho*th , df_substrate.iloc[i,-1]])
                            dist_sub.append([x/w1 , y/h1 , df_substrate.iloc[i,-1]])
                        elif dim == 3:
                            dist_substrate.append([r* phi, df_substrate.iloc[i,-1]])
                            dist_sub.append([x/w1 , y/h1 , z/z1, df_substrate.iloc[i,-1]])
    
                        
                
                    for i in range(df_sugars.shape[0]):
                        x = df_sugars.iloc[i,0]
                        y = df_sugars.iloc[i,1]
                        z = df_sugars.iloc[i,2]
                        rho = np.sqrt(x**2+y**2)
                        r = np.sqrt(x**2+y**2+z**2)
                        th = abs(np.arctan(y/x))
                        phi = np.arccos(z/r)
                        
                        if dim == 2:
                            dist_substrate.append([rho*th , df_sugars.iloc[i,-1]])
                            dist_sug.append([x/w2 , y/h2 , df_sugars.iloc[i,-1]])
                        if dim == 3:
                            dist_substrate.append([r* phi, df_sugars.iloc[i,-1]])
                            dist_sug.append([x/w2 , y/h2 , z/z2, df_sugars.iloc[i,-1]])
                            
                        
                    dist_sug = pd.DataFrame(dist_sug)
                    dist_sub = pd.DataFrame(dist_sub)
                    
                    if dim == 2:
                        for i in range(dist_sug.shape[0]):
                            x0 = dist_sug.iloc[i,0]
                            y0 = dist_sug.iloc[i,1]
                            for j in range(dist_sub.shape[0]):
                                x1 = dist_sub.iloc[j,0]
                                y1 = dist_sub.iloc[j,1]
                                r = np.sqrt((x0-x1)**2+(y0-y1)**2)
                                dist_matrix.append([r,dist_sub.iloc[j,-1][:11] + ' - ' + dist_sug.iloc[i,-1][:11]])
                                
                    if dim == 3:
                        for i in range(dist_sug.shape[0]):
                            x0 = dist_sug.iloc[i,0]
                            y0 = dist_sug.iloc[i,1]
                            z0 = dist_sug.iloc[i,2]
                            for j in range(dist_sub.shape[0]):
                                x1 = dist_sub.iloc[j,0]
                                y1 = dist_sub.iloc[j,1]
                                z1 = dist_sub.iloc[j,2]
                                r = np.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
                                dist_matrix.append([r,dist_sub.iloc[j,-1][:11] + ' - ' + dist_sug.iloc[i,-1][:11]])
                            
                        
    return param, dist_substrate, dist_matrix, dist_base


#THESE FORMULAS ARE INCORRECT!!
        
def DistanceBarPlot(param, targets,dist_substrate, dist_matrix,dist_base, stylestring):
    
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
            
    dfmelezitose = pd.DataFrame(dist_substrate,columns = ['Distance','Substrate'])    
    names = dfmelezitose.iloc[:,1].unique()
    
    NUM_COLORS = len(names)
    
    cmaps = plt.get_cmap('gist_rainbow')
    
    colours = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    
#    colours = []
#    for n in range(len(names)):
#        colours.append(cycle[n%len(cycle)])
        
#    colours = bin_colours
        
    box_pairs = []
    if stylestring == 'binary':
        for n in range(len(names)//param):
            box_pairs.append((names[n*param],names[n*param+1]))
    if stylestring == 'trinary' or stylestring == 'inter':
        for n in range(len(names)//param):
            box_pairs.append((names[n*param],names[n*param+1]))
            box_pairs.append((names[n*param],names[n*param+2]))
            box_pairs.append((names[n*param+1],names[n*param+2]))
    if stylestring == 'all':
        for n in range(len(names)):
            for k in range(len(names)):
                if n >= k:
                    pass
                else:
                    box_pairs.append((names[n],names[k]))
    if stylestring == 'one':
        for n in range(len(names)-1):
            box_pairs.append((names[0],names[n+1]))
        base = pd.DataFrame(dist_base,columns = ['Distance','Substrate'])
        dfdiff = pd.DataFrame(dist_matrix,columns = ['Distance','Substrate'])
        argmax =  dfdiff.iloc[:,1].unique()[:len(list(np.unique(targets)))//param-1][-1]
        index = dfdiff.iloc[:,1] == argmax
        argmax2 = [i for i, x in enumerate(index) if x][-1]
        dfdiff = dfdiff.iloc[:argmax2,:]
        dfdiff = pd.concat([base,dfdiff],axis=0)
    else:
        dfdiff = pd.DataFrame(dist_matrix,columns = ['Distance','Substrate'])
        
    names2 = dfdiff.iloc[:,1].unique()
    
    avgdfdiff = []
    
    for name in names2:
        indextokeep = dfdiff['Substrate']==name
        avgdfdiff.append([str(round(dfdiff['Distance'][indextokeep].mean(),2)),str(round(dfdiff['Distance'][indextokeep].std(),2)),name+ ' \n'])

    colours2 = []
    for n in range(len(names2)):
        colours2.append(cycle[n%len(cycle)])
        
#    colours2 = bin_colours
        
    box_pairs2 = []
    if stylestring == 'trinary' or stylestring == 'inter':
        for n in range(len(names2)//param):
            box_pairs2.append([names2[n*param],names2[n*param+1]])
            box_pairs2.append([names2[n*param+1],names2[n*param+2]])
            box_pairs2.append([names2[n*param],names2[n*param+2]])
        
    else:
        
        for n in range(len(names2)):
            for k in range(len(names2)):
                if n >= k:
                    pass
                else:
                    box_pairs2.append((names2[n],names2[k]))

    return avgdfdiff,box_pairs,dfmelezitose,colours, box_pairs2, names2,colours2,dfdiff




def origin_distance_plot(box_pairs, dfmelezitose,colours):
         
    y = 'Distance'  
    x = 'Substrate'
       
    test_short_name = 'spearmanr'
    pvalues = []
    for pair in box_pairs:
        data1 =  dfmelezitose.groupby(x)[y].get_group(pair[0])
        data2 =  dfmelezitose.groupby(x)[y].get_group(pair[1])
        
        
        Q1 = data1.quantile(0.25)
        Q3 = data1.quantile(0.75)
        IQR = Q3 - Q1    #IQR is interquartile range. 
        
        filter = (data1 >= Q1 - 1.5 * IQR) & (data1 <= Q3 + 1.5 *IQR)
        data1 = data1.loc[filter] 
        
        
        Q1 = data2.quantile(0.25)
        Q3 = data2.quantile(0.75)
        IQR = Q3 - Q1    #IQR is interquartile range. 
        
        filter = (data2 >= Q1 - 1.5 * IQR) & (data2 <= Q3 + 1.5 *IQR)
        data2 = data2.loc[filter] 
        
        
        if data1.shape[0] == data2.shape[0]:
            stat, p = stats.spearmanr(data1, data2)
        else:
            if data1.shape[0]>data2.shape[0]:
                dfbig = data1
                dfsmall = data2
                drop_indices = np.random.choice(dfbig.index, dfbig.shape[0]-dfsmall.shape[0], replace=False)
                dfbig2 = dfbig.drop(drop_indices)
                data1 = dfbig2
                data2 = dfsmall
                stat, p = stats.spearmanr(data1, data2)
            else:
                dfbig = data2
                dfsmall = data1
                drop_indices = np.random.choice(dfbig.index, dfbig.shape[0]-dfsmall.shape[0], replace=False)
                dfbig2 = dfbig.drop(drop_indices)
                data2 = dfbig2
                data1 = dfsmall
                stat, p = stats.spearmanr(data1, data2)
        print("Performing Bartlett statistical test for equal variances on pair:",
              pair, "stat={:.2e} p-value={:.2e}".format(stat, p))
        pvalues.append(p)
    print("pvalues:", pvalues)
    plt.figure(figsize=(9,9/1.618))
    ax = sns.boxplot(x="Substrate", y="Distance", data= dfmelezitose,palette=colours)
#    ax = sns.barplot(x="Substrate", y="Distance", data= dfmelezitose,ci="sd",capsize=.2,palette=colours)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
#    ax.set(ylim=(0,5))
    plt.title('Arc length to cluster')
    
    if np.isnan(pvalues).any():
        nan_index = [i[0] for i in np.argwhere(~np.isnan(pvalues))]
        if nan_index == []:
            pass
        else:
            add_stat_annotation(ax, data= dfmelezitose, x=x, y=y,
                                       box_pairs=[box_pairs[i] for i in nan_index],
                                       perform_stat_test=False, pvalues=[pvalues[i] for i in nan_index], 
                                       test_short_name=test_short_name,
                                       text_format='star', 
                                       verbose=2)
        
        
    else:
        add_stat_annotation(ax, data= dfmelezitose, x=x, y=y,
                                               box_pairs=box_pairs,
                                               perform_stat_test=False, 
                                               pvalues=pvalues, 
                                               test_short_name=test_short_name,
                                               text_format='star', 
                                               verbose=2)
    plt.tight_layout()
    
 
    
    
def cluster_distance_plot(dfdiff,names2,stylestring,box_pairs2,colours2):
    
    y = 'Distance'  
    x = 'Substrate'
    
    if len(names2) == 1:
        pass
    else:
            
        plt.figure(figsize=(9,9/1.618))
        plt.title('Distance between clusters')
        ax = sns.boxplot(x="Substrate", y="Distance", data= dfdiff ,palette=colours2)
#        ax = sns.barplot(x="Substrate", y="Distance", data= dfdiff ,ci="sd",capsize=.2,palette=colours2)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
#        ax.set(ylim=(0,5))
        
        if stylestring == 'binary' or stylestring == 'trinary' or stylestring == 'inter':
            test_short_name = 'spearmanr'
            pvalues = []
            for pair in box_pairs2:
                data1 =  dfdiff.groupby(x)[y].get_group(pair[0])
                data2 =  dfdiff.groupby(x)[y].get_group(pair[1])
                if data1.shape[0] == data2.shape[0]:
                    stat, p = stats.spearmanr(data1, data2)
                else:
                    if data1.shape[0]>data2.shape[0]:
                        dfbig = data1
                        dfsmall = data2
                        drop_indices = np.random.choice(dfbig.index, dfbig.shape[0]-dfsmall.shape[0], replace=False)
                        dfbig2 = dfbig.drop(drop_indices)
                        data1 = dfbig2
                        data2 = dfsmall
                        stat, p = stats.spearmanr(data1, data2)
                    else:
                        dfbig = data2
                        dfsmall = data1
                        drop_indices = np.random.choice(dfbig.index, dfbig.shape[0]-dfsmall.shape[0], replace=False)
                        dfbig2 = dfbig.drop(drop_indices)
                        data2 = dfbig2
                        data1 = dfsmall
                        stat, p = stats.spearmanr(data1, data2)
                print("Performing Bartlett statistical test for equal variances on pair:",
                      pair, "stat={:.2e} p-value={:.2e}".format(stat, p))
                pvalues.append(p)
                print("pvalues:", pvalues)
                
            if np.isnan(pvalues).any():
                nan_index = [i[0] for i in np.argwhere(~np.isnan(pvalues))]
                if nan_index == []:
                    pass
                else:
                    add_stat_annotation(ax, data= dfdiff, x=x, y=y,
                                                   box_pairs=[box_pairs2[i] for i in nan_index],
                                                   perform_stat_test=False,
                                                   pvalues=[pvalues[i] for i in nan_index], 
                                                   test_short_name=test_short_name,
                                                   text_format='star', verbose=2)
            
                
            else:       
                
                add_stat_annotation(ax, data= dfdiff, x=x, y=y,
                                           box_pairs=box_pairs2,
                                           perform_stat_test=False, 
                                           pvalues=pvalues, 
                                           test_short_name=test_short_name,
                                           text_format='star', 
                                           verbose=2)
        
        elif stylestring == 'one':
            test_short_name = 'spearmanr'
            pvalues = []
            for pair in box_pairs2[:len(names2)-1]:
                data1 =  dfdiff.groupby(x)[y].get_group(pair[0])
                data2 =  dfdiff.groupby(x)[y].get_group(pair[1])
                if data1.shape[0] == data2.shape[0]:
                    stat, p = stats.spearmanr(data1, data2)
                else:
                    if data1.shape[0]>data2.shape[0]:
                        dfbig = data1
                        dfsmall = data2
                        drop_indices = np.random.choice(dfbig.index, dfbig.shape[0]-dfsmall.shape[0], replace=False)
                        dfbig2 = dfbig.drop(drop_indices)
                        data1 = dfbig2
                        data2 = dfsmall
                        stat, p = stats.spearmanr(data1, data2)
                    else:
                        dfbig = data2
                        dfsmall = data1
                        drop_indices = np.random.choice(dfbig.index, dfbig.shape[0]-dfsmall.shape[0], replace=False)
                        dfbig2 = dfbig.drop(drop_indices)
                        data2 = dfbig2
                        data1 = dfsmall
                        stat, p = stats.spearmanr(data1, data2)
                print("Performing Bartlett statistical test for equal variances on pair:",
                      pair, "stat={:.2e} p-value={:.2e}".format(stat, p))
                pvalues.append(p)
                print("pvalues:", pvalues)
                
            if np.isnan(pvalues).any():
                nan_index = [i[0] for i in np.argwhere(~np.isnan(pvalues))]
                if nan_index == []:
                    pass
                else:
                    add_stat_annotation(ax, data= dfdiff, x=x, y=y,
                                                   box_pairs=[box_pairs2[i] for i in nan_index][:len(names2)-1],
                                                   perform_stat_test=False, 
                                                   pvalues=[pvalues[i] for i in nan_index], 
                                                   test_short_name=test_short_name,
                                                   text_format='star', 
                                                   verbose=2)
                
                
            else:
                add_stat_annotation(ax, data= dfdiff, x=x, y=y,
                                           box_pairs=box_pairs2[:len(names2)-1],
                                           perform_stat_test=False, 
                                           pvalues=pvalues, 
                                           test_short_name=test_short_name,
                                           text_format='star', 
                                           verbose=2)
        elif stylestring == 'all':
            test_short_name = 'spearmanr'
            pvalues = []
            for pair in box_pairs2:
                data1 =  dfdiff.groupby(x)[y].get_group(pair[0])
                data2 =  dfdiff.groupby(x)[y].get_group(pair[1])
                if data1.shape[0] == data2.shape[0]:
                    stat, p = stats.spearmanr(data1, data2)
                else:
                    if data1.shape[0]>data2.shape[0]:
                        dfbig = data1
                        dfsmall = data2
                        drop_indices = np.random.choice(dfbig.index, dfbig.shape[0]-dfsmall.shape[0], replace=False)
                        dfbig2 = dfbig.drop(drop_indices)
                        data1 = dfbig2
                        data2 = dfsmall
                        stat, p = stats.spearmanr(data1, data2)
                    else:
                        dfbig = data2
                        dfsmall = data1
                        drop_indices = np.random.choice(dfbig.index, dfbig.shape[0]-dfsmall.shape[0], replace=False)
                        dfbig2 = dfbig.drop(drop_indices)
                        data2 = dfbig2
                        data1 = dfsmall
                        stat, p = stats.spearmanr(data1, data2)
                print("Performing Bartlett statistical test for equal variances on pair:",
                      pair, "stat={:.2e} p-value={:.2e}".format(stat, p))
                pvalues.append(p)
                print("pvalues:", pvalues)
                
            if np.isnan(pvalues).any():
                nan_index = [i[0] for i in np.argwhere(~np.isnan(pvalues))]
                if nan_index == []:
                    pass
                else:
                    add_stat_annotation(ax, data= dfdiff, x=x, y=y,
                                                   box_pairs=[box_pairs2[i] for i in nan_index],
                                                   perform_stat_test=False, 
                                                   pvalues=[pvalues[i] for i in nan_index], 
                                                   test_short_name=test_short_name,
                                                   text_format='star', 
                                                   verbose=2)
                
                
            else:
                add_stat_annotation(ax, data= dfdiff, x=x, y=y,
                                           box_pairs=box_pairs2,
                                           perform_stat_test=False, pvalues=pvalues, 
                                           test_short_name=test_short_name,
                                           text_format='star', 
                                           verbose=2)
                
    
    #    avgdfdiff = pd.DataFrame(avgdfdiff,columns=['Avg Distance','Substrate'])
    #    sns.lineplot(data = avgdfdiff['Avg Distance'], ax=ax3,color='r')
        plt.tight_layout()
        
       

def density_plot(dfdiff,colours2,names2): 
    plt.figure(figsize=(9,9/1.618))
    plt.title('Distance between clusters density plot')
    ax = sns.histplot(dfdiff,x="Distance",hue="Substrate",element="poly",palette=colours2,label=names2)
    h,l = ax.get_legend_handles_labels()
    ax.legend(handles=h, labels=list(names2), ncol = len(names2)//6+1)
    
#dim = 3
    
def LoadingsPlot(wavenumber,pca,dim):
    plt.figure(figsize=(9,9/1.618))
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_))
    if dim == 2:
        loadings = loadings.iloc[:,:2]
        loadings.columns = ['LPC1','LPC2']
        plt.plot(wavenumber,savgol_filter(loadings['LPC1'],11,3),label='LPC1')
        plt.plot(wavenumber,savgol_filter(loadings['LPC2'],11,3),ls='--',label='LPC2')
    elif dim == 3:
        loadings = loadings.iloc[:,:3]
        loadings.columns = ['LPC1','LPC2','LPC3']
        plt.plot(wavenumber,savgol_filter(loadings['LPC1'],11,3),label='LPC1')
        plt.plot(wavenumber,savgol_filter(loadings['LPC2'],11,3),ls='--',label='LPC2')
        plt.plot(wavenumber,savgol_filter(loadings['LPC3'],11,3),ls=':',label='LPC3')
    plt.legend(loc='best',frameon=False)
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Loadings')
    plt.show()
        

#####################################################################################
    ##### 3D PCA PLOT AND LOADINGS ##################################################
    #################################################################################

#http://kylebarbary.com/nestle/examples/plot_ellipsoids.html

def plot_ellipsoid_3d(ell, ax,color):
    """Plot the 3-d Ellipsoid ell on the Axes3D ax."""

    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    z = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    x = np.outer(np.ones_like(u), np.cos(v))

    # transform points to ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            x[i,j], y[i,j], z[i,j] = ell.ctr + np.dot(ell.axes,[x[i,j],y[i,j],z[i,j]])

    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=color, alpha=0.2)

#ax = plt.axes(projection = '3d')

def PCAPlot3D(df,target,info,ax,flag):    
    positionz = []

    if flag == 0:
    
#        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
        

        ax.view_init(30,-45)
        ax.set_xlabel(str(info[0]),linespacing=5)
        ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))           
        ax.set_ylabel(str(info[1]),linespacing=5)
        ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))            
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(str(info[2]),rotation=90,linespacing=5)
        ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
        
#        colours = []
#        
#        for i in range(len(list(np.unique(target)))):
#            colours.append(cycle[i%len(cycle)])
        
        NUM_COLORS = len(list(np.unique(target)))
        
        cmaps = plt.get_cmap('gist_rainbow')
        
        colours = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
            
#        colours = bin_colours
            
        for label,color in zip(list(np.unique(target)),colours):
            indicesToKeep = df['sample'] == label
    
            new_x = df.loc[indicesToKeep, 'principal component 1']
            new_y = df.loc[indicesToKeep, 'principal component 2']
            new_z = df.loc[indicesToKeep, 'principal component 3']
    
            ax.scatter(new_x, new_y,new_z, s=100, alpha=0.6, label=label,marker='o',color=color)
    
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(loc='best',frameon=False)
            
            
            positionz.append([[(new_x.max()-new_x.min())/2,(new_y.max()-new_y.min())/2,(new_z.max()-new_z.min())/2],
           (new_x.max()-new_x.min()),
           (new_y.max()-new_y.min()),
           (new_z.max()-new_z.min()),label])
            
            plt.show()
        
        
    else:
        
#        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
            
        
        ax.view_init(30,-45)
        ax.set_xlabel(str(info[0]),linespacing=5)
        ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))           
        ax.set_ylabel(str(info[1]),linespacing=5)
        ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))            
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(str(info[2]),rotation=90,linespacing=5)
        ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
        
#        positionz = []
#        colours = []
#        
#        for i in range(len(list(np.unique(target)))):
#            colours.append(cycle[i%len(cycle)])
        
        
        NUM_COLORS = len(list(np.unique(target)))
        
        cmaps = plt.get_cmap('gist_rainbow')
        
        colours = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
        
            
#        colours = bin_colours
                
        for label,color in zip(list(np.unique(target)),colours):
            indicesToKeep = df['sample'] == label
            new_x = df.loc[indicesToKeep, 'principal component 1']
            new_y = df.loc[indicesToKeep, 'principal component 2']
            new_z = df.loc[indicesToKeep, 'principal component 3']
            
            if df.loc[indicesToKeep].shape[0] < 2:
                
                ax.scatter(new_x, new_y,new_z, s=100, alpha=0.6, color=color ,label=label,marker='o')
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(loc='best',frameon=False)
            
            else:
            
                gmmf = GaussianMixture(n_components=1).fit(pd.concat([df.loc[indicesToKeep, 'principal component 1'],
                                      df.loc[indicesToKeep, 'principal component 2']
                                   , df.loc[indicesToKeep, 'principal component 3']],axis=1).values)  
                
                centers = gmmf.means_
                U, s, Vt = np.linalg.svd(gmmf.covariances_)
                widths, heights,zeds = 2 * np.sqrt(s[0])  
                
                positionz.append([centers[0],widths,heights,zeds,label])
    
                range1 = (new_x < centers[0][0]+np.sqrt(widths/2)) & (new_x > centers[0][0]-np.sqrt(widths/2))
                range2 = (new_y < centers[0][1]+np.sqrt(heights/2)) & (new_y > centers[0][1]-np.sqrt(heights/2))
                range3 = (new_z < centers[0][2]+np.sqrt(zeds/2)) & (new_z > centers[0][2]-np.sqrt(zeds/2))
                       
                npoints = len(new_x[range1 & range2 & range3])
                
                A = np.array([[1/np.sqrt(widths/2),0,0],
                  [0,1/np.sqrt(heights/2),0],
                  [0,0,1/np.sqrt(zeds/2)]])
        
                ell_gen = nestle.Ellipsoid([centers[0][0], centers[0][1], centers[0][2]], np.dot(A.T, A))
                
                concat = pd.concat([new_x[range1 & range2 & range3], new_y[range1 & range2 & range3],new_z[range1 & range2 & range3]],axis=1).values
                
    #            points = ell_gen.samples(npoints)
                pointvol = ell_gen.vol / npoints
                
                # Find bounding ellipsoid(s)
                if concat.size == 0 or np.isinf(pointvol):
                    
                    ax.scatter(new_x, new_y ,new_z, s=100, alpha=0.6, color=color ,label=label,marker='o')
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = OrderedDict(zip(labels, handles))
                    plt.legend(loc='best',frameon=False)
                
                else:
                    
                    ells = nestle.bounding_ellipsoid(concat, pointvol)
                           
                    ax.scatter(new_x[range1 & range2 & range3], new_y[range1 & range2 & range3],new_z[range1 & range2 & range3], s=100, alpha=0.6, color=color ,label=label,marker='o')
        #            for ell in ells:
                    plot_ellipsoid_3d(ells, ax, color)
    
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = OrderedDict(zip(labels, handles))
                    plt.legend(loc='best',frameon=False)
            
        plt.show()
        
    return positionz




def find_local_max(signal):
    dim = signal.shape[0]
    ans = np.zeros(dim)
    for i in range(dim):
        dxf = signal[i - 1] - signal[i]
        dxb = signal[i - 1] - signal[i - 2]
        ans[i] = 1.0 if dxf > 0 and dxb > 0 else 0.0
    return ans > 0.5


def fit_region(wavenumber,pos,tol = 10):
    min_l = min(wavenumber.index.tolist())
    max_l = max(wavenumber.index.tolist())
    index = wavenumber.index[wavenumber==pos].tolist()[0]
    index1 = index-tol
    index2 = index+tol
    if index1<min_l:
        index1 = 0
    if index2>max_l:
        index2 = max_l
        
    return wavenumber[index1:index2]

def linear_fit(x,a,b):
    return a*x+b

def power_law(x, a, b):
    return a*np.power(x, b)

def _1Lorentzian(x, amp, cen, wid):
    return amp*wid**2/((x-cen)**2+wid**2)

# From LineLaplacianBuilder
def get_line_laplacian_eigen(n):
    assert n > 1
    eigen_vectors = np.zeros([n, n])
    eigen_values = np.zeros([n])

    for j in range(1, n + 1):
        theta = np.pi * (j - 1) / (2 * n)
        sin = np.sin(theta)
        eigen_values[j - 1] = 4 * sin * sin
        if j == 0:
            sqrt = 1 / np.sqrt(n)
            for i in range(1, n + 1):
                eigen_vectors[i - 1, j - 1] = sqrt
        else:
            for i in range(1, n + 1):
                theta = (np.pi * (i - 0.5) * (j - 1)) / n
                math_sqrt = np.sqrt(2.0 / n)
                eigen_vectors[i - 1, j - 1] = math_sqrt * np.cos(theta)
    return eigen_vectors, eigen_values


def smooth3(t, signal):
    signal_i = signal[0]
    dim = signal.shape[0]
    U, s = get_line_laplacian_eigen(dim)
    return matrix_exp_eigen(U, -s, t, signal)-(matrix_exp_eigen(U, -s, t, signal)[0]-signal_i)

from matplotlib import gridspec   
import os.path
from scipy.optimize import curve_fit         
from adjustText import adjust_text
import re

#master = 'MASTER.csv'
def clean_master(master):
#    df = file
    df = pd.read_csv(master, sep = ';')
    df = df.drop_duplicates()
    index = df.columns[0] != df.iloc[:,0]
    df = df[index]
    
    df.to_csv('MASTERKEY.csv', sep=';', index=False)
    
    return df


#peak
def clean_master_df(peak):
    df = peak.drop_duplicates()
    index = df.columns[0] != df.iloc[:,0]
    df = df[index]
    
    return df
 


# files = [
#     r'/home/newuser/Desktop/saliva/saliva data/milk/saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
#          r'/home/newuser/Desktop/saliva/saliva data/milk/milk_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt']

# smooth = 7
# asymmetry_param = 0.05
# smoothness_param = 1000000
# max_iters = 10
# conv_thresh =0.00001   




def peakfinder(files,smooth,asymmetry_param,smoothness_param ,max_iters,conv_thresh):
    zscore = 0
    inv = 0

    peak_vector = []
    max_value = []
    spectra = []
    err = []
    
#    if len(files)==1:
#        files = [files]
        
    for file in files:
#        print(file)

        fig = plt.figure(figsize=(12,12/1.618))
        gs = gridspec.GridSpec(4,1, height_ratios=[1,1,0.25,1])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])
        ax3 = fig.add_subplot(gs[3])
        ax4 = fig.add_subplot(gs[0])
    
        wavenumber, intensity, n_points, n_raman, label,flag = ImportData(file)
       
        intensity_b = Baseline(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)
        max_value.append(intensity_b.max())
        
        spectra.append(Normalize(intensity_b,zscore,inv))
        
#        smoothed_signal = Normalize(intensity_b,zscore,inv)
#        smoothed_signal = Normalize(smoothed_signal,zscore,inv)
        
        smoothed_signal = smooth2(smooth,  intensity_b)
        smoothed_signal = Normalize(smoothed_signal,zscore,inv)
       
        local_max_index = find_local_max(smoothed_signal)
    
        stem = np.zeros(wavenumber.shape)
        stem[local_max_index] = 1  
        
        lorentz = []

        
        for wav in wavenumber[local_max_index]:
    
            reg = fit_region(wavenumber,wav,10)
            
            try:
                popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                                reg, 
                                                                smoothed_signal[reg.index],
                                                                p0=[smoothed_signal[reg.index].max(), wav,  2/(np.pi*smoothed_signal[reg.index].max())])
                
                perr_1lorentz = np.sqrt(np.diag(pcov_1lorentz))
        
                pars_1 = popt_1lorentz[0:3]
                
                if (abs(pars_1[-1])>100) | (pars_1[0]<0):
                    pass
                else:
                    
                    lorentz_peak_1 = _1Lorentzian(reg, *pars_1)
                    
                    ax4.plot(wavenumber,_1Lorentzian(wavenumber,*popt_1lorentz))
                    ax4.fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),
                                         _1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
                    
                    
        
                    lorentz.append(_1Lorentzian(wavenumber,*popt_1lorentz))
                    peak_vector.append([label]+popt_1lorentz.tolist()+['major'])
                    err.append(perr_1lorentz[0])
            
            except:
                pass
     
        lorentz_all = Normalize(Baseline(pd.DataFrame(lorentz).sum() ,asymmetry_param, smoothness_param, max_iters, conv_thresh),zscore,inv)
        residual_1lorentz = smoothed_signal - lorentz_all 
        

        
        local_max_index2 = find_local_max(residual_1lorentz.values)
        stem2 = np.zeros(wavenumber.shape)
        stem2[local_max_index2] = 1 
    
        local_max_index3 = find_local_max(-residual_1lorentz.values)
        stem3 = np.zeros(wavenumber.shape)
        stem3[local_max_index3] = 1 
        

        ax4.plot(wavenumber, smoothed_signal ,'o',color='orange')
        ax4.plot(wavenumber[local_max_index],smoothed_signal[local_max_index],'x',color='r')
        
        
        lorentz2 = []
        lorentz_neg = []
        
        for wav in wavenumber[local_max_index2]:
            reg = fit_region(wavenumber,wav,10)
            if residual_1lorentz[reg[reg==wav].index].tolist() == []:
                reg = round(reg)
                wav = round(wav+1)
            elif (residual_1lorentz[reg[reg==wav].index].values[0] > 0):

                try:
                    popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                                    reg, 
                                                                    residual_1lorentz[reg.index],
                                                                    p0=[residual_1lorentz[reg.index].max(), wav, 2/(np.pi*residual_1lorentz[reg.index].max())])
                    
                    perr_1lorentz = np.sqrt(np.diag(pcov_1lorentz))
            
                    pars_1 = popt_1lorentz[0:3]
                    
                    if (abs(pars_1[-1])>100) | (pars_1[0]<0):
                        pass
                    else:
                    
                        lorentz_peak_1 = _1Lorentzian(reg, *pars_1) 
                        
                        
                        ax1.plot(wavenumber,_1Lorentzian(wavenumber,*popt_1lorentz))
                        ax1.fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),
                                         _1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
                        
                        lorentz2.append(_1Lorentzian(wavenumber,*popt_1lorentz))
                        peak_vector.append([label]+popt_1lorentz.tolist()+['minor positive'])
                        err.append(perr_1lorentz[0])
                
                except:
                    pass
                
        ax1.plot(wavenumber,residual_1lorentz,'o',color='b')
        ax1.plot(wavenumber[local_max_index2],residual_1lorentz[local_max_index2],'x',color='orange')
        
#        ax4.plot(wavenumber,-residual_1lorentz,'o')
        ax1.plot(wavenumber[local_max_index3],residual_1lorentz[local_max_index3],'x',color='orange')
                        
        for wav in wavenumber[local_max_index3]:
            reg = fit_region(wavenumber,wav,10)
            if residual_1lorentz[reg[reg==wav].index].tolist() == []:
                reg = round(reg)
                wav = round(wav+1)
            elif (-residual_1lorentz[reg[reg==wav].index].values[0] > 0):
    
                try:
                    popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                                    reg, 
                                                                    -residual_1lorentz[reg.index],
                                                                    p0=[-residual_1lorentz[reg.index].min(), wav,  2/(np.pi*-1*residual_1lorentz[reg.index].max())])
                    perr_1lorentz = np.sqrt(np.diag(pcov_1lorentz))
        
                    pars_1 = popt_1lorentz[0:3]
                    
                    if (abs(pars_1[-1])>100) | (pars_1[0]<0):
                        pass
                    else:
                    
                        lorentz_peak_1 = _1Lorentzian(reg, *pars_1) 
        
                        
                        ax1.plot(wavenumber,-_1Lorentzian(wavenumber,*popt_1lorentz))
                        ax1.fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),
                                         -_1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
                        
                        lorentz_neg.append(_1Lorentzian(wavenumber,*popt_1lorentz))
        
                        peak_vector.append([label]+[a*b for a,b in zip(popt_1lorentz.tolist(),[-1,1,1])]+['minor negative'])
                        err.append(perr_1lorentz[0])
                except:
                    pass
                
                          
                
            
        #not normalized because the residue is smaller than 1, still dont know if its better to use .sum() or .mean()???
        lorentz_all2 = Baseline(pd.DataFrame(lorentz2).sum() ,asymmetry_param, smoothness_param, max_iters, conv_thresh)
        
        lorentz_all3 = Baseline(pd.DataFrame(lorentz_neg).sum() ,asymmetry_param, smoothness_param, max_iters, conv_thresh)
        
        residual_1lorentz2 = residual_1lorentz - lorentz_all2 
        residual_1lorentz3 = residual_1lorentz - lorentz_all2 + lorentz_all3
        
        ax2.plot(wavenumber, residual_1lorentz,'o',color='b',alpha=0.4)
#        ax2.plot(wavenumber, residual_1lorentz2,'o',color='g',alpha=0.4)
        ax2.plot(wavenumber, residual_1lorentz3,'o',color='r',alpha=1)
    
        ax3.plot(wavenumber,Normalize(intensity_b,zscore,inv),'o',color='orange')
        ax3.plot(wavenumber, Normalize(lorentz_all+lorentz_all2-lorentz_all3,zscore,inv),color='C0',lw=2)
    
    
    for peak, e in zip(peak_vector,err):
        peak.append(e)
    
    peak = pd.DataFrame(peak_vector,columns=['label','height','center','width','importance','err'] )
    
    spectra.append(wavenumber)  
    
    spectra = pd.DataFrame(spectra).T
#    print(spectra)
    spectra.columns = peak['label'].unique().tolist()+['wavenumber']
    
    new_peak = []
    new_err = []
    
    for i in range(spectra.shape[1]-1):
        spectra.iloc[:,i] = spectra.iloc[:,i]*max_value[i]
        index = spectra.columns[i] == peak['label']
        new_peak.append(peak['height'].loc[index]*max_value[i])
        new_err.append(peak['err'].loc[index]*max_value[i])
        
    peak['Intensity (a.u.)'] = pd.DataFrame(np.concatenate([np.stack(a) for a in new_peak]))
    peak['err'] = pd.DataFrame(np.concatenate([np.stack(a) for a in new_err]))
        
    for lab in peak['label'].unique():
        peak[peak['label']==lab].to_csv('key'+lab+'.csv',sep=';', index=False)

#    peak.to_csv('key'+label+'.csv',sep=';', index=False)
    
    if os.path.isfile('MASTERKEY.csv'):
        peak.to_csv('MASTERKEY.csv', sep=';', index=False,mode='a')
        clean_master('MASTERKEY.csv')
    else:
        peak.to_csv('MASTERKEY.csv', sep=';', index=False)
        
    return peak,spectra
    
#convert string in file name related to concentration to actual values
#file = 'MASTER.csv' 
#file = files
    
def clean_up_rep(peak):
    df = clean_master_df(peak)
#    df = pd.read_csv(file, sep = ';')
    
#    z = np.abs(stats.zscore(df['center']))
#    df = df[(z < 1)]
    
#    z2 = np.abs(stats.zscore(df['Intensity (a.u.)']))
#    df = df[(z2<1)]

    string = ['pM','nM','uM','mM','M']
    mult = [10**-12,10**-9,10**-6,10**-3,1] 
    
    conc_s = []
    dil = df['label'].values
    

    for d in dil:
        for s in string:
            #solves for decimals
#            match = re.findall(r"\d+\.\d*"+s,d)
            #solves for integers
#            match2 =  re.findall(r"[-+]?\d*\.\d+|\d+?"+s,d)
            match3 = re.findall(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"+s,d)
#            print(s,match3)
            if match3:
                conc_s.append(match3[0][0]+s)

#                conc_s.append(match.group(0))
    if not conc_s:
        arbitrary = []
        i=0
        for lab in df['label'].unique().tolist():
            index = df['label'] == lab
            arbitrary.append([i]*sum(index))
            i = i+1
        
        arbitrary = [item for sublist in arbitrary for item in sublist]
        df['Concentration[M]'] = arbitrary

    else:
        
        conc = []
        
        for c in conc_s:
            for s,m in zip(string,mult):
                try:
                    conc.append(float(c.split(s)[0])*m)
                except:
                    pass
        if not conc:
            pass
        else:
            df['Concentration[M]'] = conc

    return df ,conc_s

#clusters close peaks

def cluster(data, maxgap):
    data.sort()
    groups = [[data[0]]]
    for item in data[1:]:
        val = abs(item - groups[-1][-1])
        if val <= maxgap:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


#convert repetitive peaks into their average and keep the strings

def cleaner_df(dataset):
    new_df = []
    for i in range(dataset.shape[1]):
        if type(dataset.iloc[:,i].values[0]) is str:
            new_df.append(dataset.iloc[:,i].values[0])
        else:
            new_df.append(dataset.iloc[:,i].mean())
    return pd.DataFrame(new_df,index = dataset.columns).T


#remove repeatitive peaks

def clean_df(df):
   
    for label in df['label'].unique().tolist():
        index = df['label'] == label
        cluster_l = cluster(df['center'][index].values,7)
        for c in cluster_l:
            if len(c)>1:
                dataset = pd.concat([df['center']==d for d in c],axis=1).any(axis=1)
                clean_dataset = cleaner_df(df[dataset])
                df = df.drop(index=df[dataset].index)
                df = df.append(clean_dataset).reset_index(drop=True)
    return df


import matplotlib as mpl


#file = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\MASTER.csv'

#file = peak

def exponential(x,a,b):
    return a*np.exp(x*b)
    
#
#number = 0

def plot_dilution(peak,spectra,number):
    
#    dil_peaks = []
    
    if number == 0:

        df,conc_s = clean_up_rep(peak)
        wavenumber = spectra.iloc[:,-1]
    
        if not conc_s:
            larg_conc = df['Intensity (a.u.)'].max()
            index1 =  df['Concentration[M]'] == df['Concentration[M]'][df['Intensity (a.u.)']== larg_conc].values[0]
        
        else:
                        
            larg_conc = df['Concentration[M]'].max()
            index1 = df['Concentration[M]'] == larg_conc
        
    
        index2 = df['Intensity (a.u.)'] >0
        index3 = (wavenumber.min()<=df['center']) & (df['center']<=wavenumber.max())
        index4 = df['importance'] == 'major'
        
        index = index1 & index2 & index3 & index4
#        index = index1
        #most important peaks,hight concentration, major, above zero and between the wavenumber
        
    #    maxh = df[index]['Intensity (a.u.)']
        center_maxh = df[index]['center']
        toplot = []
        
        bestmatch = []  
        
        
        plt.figure(figsize=(9,9/1.618))
        for label in spectra.iloc[:,:-1].columns.tolist():
#            col = label == spectra.iloc[:,:-1].columns
            col = label == spectra.columns
            plt.plot(spectra.iloc[:,-1],spectra.iloc[:,col],label = label[:20])
        plt.vlines(center_maxh,0,spectra.max().max(),ls='--',color='gray')
        plt.legend(loc='best',frameon=False)
        plt.xlabel('Raman shift $(cm^{-1})$')
        plt.ylabel('Intensity (a. u.)')
        plt.show()
        
        df = df[index2]
        
    #    if spectra.shape[1]>6:
    #        df = df[index4]
        
        for cent in center_maxh.tolist():
            for sample in df['label'].unique().tolist():
                index1 = df['label'] == sample
                index2 = (int(cent)-7 < df['center'][index1] ) & (df['center'][index1]<int(cent)+7)
    #            index3 = df['importance'] == 'major'
                index = (index1 & index2)
                try:
                    toplot.append([cent,df['Concentration[M]'][index1].mean(),df['Intensity (a.u.)'][index].values[0],sample[:20]])
                except:
                    toplot.append([cent,df['Concentration[M]'][index1].mean(),0,sample[:20]])
    
      
            
        toplot = pd.DataFrame(toplot, columns = ['Center (cm-1)','Concentration [M]','Intensity (a.u.)','label'])  
        toplot = toplot.sort_values(by='Intensity (a.u.)').reset_index(drop=True)
        toplot.to_csv('dilutions_'+toplot['label'].iloc[-1]+'.csv',index=False)  
        
        toplotmax = toplot['Intensity (a.u.)'].max()
        toplotmin = toplot['Intensity (a.u.)'][toplot['Intensity (a.u.)']>0].min()
        
        fig, ax = plt.subplots(figsize=(9,9/1.618))
        
        cmap = mpl.cm.get_cmap('cool')
        
        colors = [cmap(x) for x in np.linspace(0,1,len(sorted(toplot['Center (cm-1)'].unique())))]
        
        texts = []
        for cent,color in zip(toplot['Center (cm-1)'].unique(),colors):
    #        print(toplot['label'][index].tolist())
            index =  toplot['Center (cm-1)'] == cent
            
            #NO IDEA WHY I PUT THE -1 INDEXATION...???
            
            x =  toplot['Concentration [M]'][index].values
            y = toplot['Intensity (a.u.)'][index].values
    #        plt.plot(len(x),len(y),'o')
            label = toplot['label'][index].tolist()
    #        print(x,y,label)
            
            if (y==toplotmax).any() or (y==toplotmin).any():
                alpha = 1
            else:
                alpha = 0.2
            
            if len(df['label'].unique().tolist()) == 1:
                ax.plot(x,y,'o',color=color,alpha=alpha)
                ax.set_xscale('log')
                ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
                texts.append(ax.text(x[-1], y[-1],str(round(cent,1))+' cm-1'))
                ax.set_xlim(x.min()/5,x.max()*5)
    
                if not conc_s:
                    my_xticks = toplot['label'].unique()
                    ax.set_xticks(toplot['Concentration [M]'].unique(), my_xticks)
                    plt.xlabel('arb. unit')
                else:
                    plt.xlabel('Concentration (M)')
                plt.ylabel('Intensity (a.u.)')
                plt.show()
                
            else:
                
                if y[-1] >= toplot['Intensity (a.u.)'].mean():
        
                    xnew = np.linspace(x.min(),x.max(),10000)
                    
                    if not conc_s:
                        fit = exponential
                        rsup=0.8
                        plt.xlabel('arb. unit')
                        ax.set_xscale('linear')
                        ax.set_xlim(x.min()-1,x.max()+1)
                        my_xticks = toplot['label'].unique()
                        plt.xticks(toplot['Concentration [M]'].unique(), my_xticks)
    #                    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
                    else:
                        fit = power_law
                        rsup = 0.95
                        plt.xlabel('Concentration (M)')
                        ax.set_xscale('log')
                        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
                        ax.set_xlim(x.min()/5,x.max()*5)
                        
                    try:
                            
                        pars, cov = curve_fit(f=fit, xdata=x, ydata=y, p0=[1, 1])
                        stdevs = np.sqrt(np.diag(cov))
                        res = y - fit(x, *pars)
                        ss_res = np.sum(res**2)
                        ss_tot = np.sum((y-np.mean(y))**2)
                        r_squared = 1 - (ss_res / ss_tot)
                        
                        if r_squared>rsup:
    
                            ax.plot(x, y,'o', color = color,alpha=alpha)
                            ax.plot(xnew, fit(xnew, *pars), linestyle='--', linewidth=2,color=color,alpha=alpha)
                            texts.append(ax.text(x[-1], y[-1],str(round(cent,1))+' cm-1'))
                            
#                            for sx,sy in zip(my_xticks,y):
#                                dil_peaks.append([cent,sy,sx])
                            
                            if alpha == 1:
                                if not conc_s:
                                    bestmatch = 'I(x)='+str(round(pars[0],2))+'(+/-'+str(round(stdevs[0],2))+')*exp(x*'+str(round(pars[1],2))+'(+/-'+str(round(stdevs[1],2))+'))'
                                    ax.text(x.min()*0.9, y.max()*0.9,bestmatch)
                                else:
                                    bestmatch = 'I(x)='+str(round(pars[0],2))+'(+/-'+str(round(stdevs[0],2))+')*x$^{'+str(round(pars[1],2))+'(+/-'+str(round(stdevs[1],2))+')}$'
                                    ax.text(x.min()*0.9, y.max()*0.9,bestmatch)
                                    
        
                            
                    except:
                        pass
                    
                plt.ylabel('Intensity (a.u.)')
                plt.show()
                
            adjust_text(texts, autoalign ='xy')  
    else:
        
        df,conc_s = clean_up_rep(peak)
        wavenumber = spectra.iloc[:,-1]
    
        if not conc_s:
            larg_conc = df['Intensity (a.u.)'].max()
            index1 =  df['Concentration[M]'] == df['Concentration[M]'][df['Intensity (a.u.)']== larg_conc].values[0]
        
        else:
                        
            larg_conc = df['Concentration[M]'].max()
            index1 = df['Concentration[M]'] == larg_conc
        
    
        index2 = df['Intensity (a.u.)'] >0
        index3 = (wavenumber.min()<=df['center']) & (df['center']<=wavenumber.max())
        
        
        
        index4 = df['importance'] == 'major'
        
        index = index1 & index2 & index3 & index4
        
        #most important peaks,hight concentration, major, above zero and between the wavenumber
        
    #    maxh = df[index]['Intensity (a.u.)']
        center_maxh = df[index]['center']    
        toplot = []
        
        bestmatch = []  
        
        center_numb = center_maxh[(number-5<center_maxh)&(center_maxh<number+5)]
        
        plt.figure(figsize=(9,9/1.618))
        for label in spectra.iloc[:,:-1].columns.tolist():
#            col = label == spectra.iloc[:,:-1].columns
            col = label == spectra.columns
            plt.plot(spectra.iloc[:,-1],spectra.iloc[:,col],label = label[:20])
        plt.vlines(center_numb,0,spectra.max().max(),ls='--',color='gray')
        plt.legend(loc='best',frameon=False)
        plt.xlabel('Raman shift $(cm^{-1})$')
        plt.ylabel('Intensity (a. u.)')
        plt.show()
        
    #    df = df[index2]
        
    #    if spectra.shape[1]>6:
    #        df = df[index4]
        
        for cent in center_numb:
            for sample in df['label'].unique().tolist():
                index1 = df['label'] == sample
                index2 = (int(cent)-7 < df['center'][index1] ) & (df['center'][index1]<int(cent)+7)
    #            index3 = df['importance'] == 'major'
                index = (index1 & index2)
                try:
                    toplot.append([cent,df['Concentration[M]'][index1].mean(),df['Intensity (a.u.)'][index].values[0],sample[:20]])
                except:
                    toplot.append([cent,df['Concentration[M]'][index1].mean(),0,sample[:20]])
    
        
        toplot = pd.DataFrame(toplot, columns = ['Center (cm-1)','Concentration [M]','Intensity (a.u.)','label'])  
        toplot = toplot.sort_values(by='Intensity (a.u.)').reset_index(drop=True)
        toplot.to_csv('dilutions_'+toplot['label'].iloc[-1]+'.csv',index=False) 
        
        toplotmax = toplot['Intensity (a.u.)'].max()
        toplotmin = toplot['Intensity (a.u.)'][toplot['Intensity (a.u.)']>0].min()
    
    
        fig, ax = plt.subplots(figsize=(9,9/1.618))
        
        cmap = mpl.cm.get_cmap('cool')
        
        colors = [cmap(x) for x in np.linspace(0,1,len(sorted(toplot['Center (cm-1)'].unique())))]
        
        texts = []
        for cent,color in zip(toplot['Center (cm-1)'].unique(),colors):
    #        print(toplot['label'][index].tolist())
            index =  toplot['Center (cm-1)'] == cent
            
            #NO IDEA WHY I PUT THE -1 INDEXATION...???
            
            x =  toplot['Concentration [M]'][index].values
            y = toplot['Intensity (a.u.)'][index].values
    #        plt.plot(len(x),len(y),'o')
            label = toplot['label'][index].tolist()
    #        print(x,y,label)
            
            if (y==toplotmax).any() or (y==toplotmin).any():
                alpha = 1
            else:
                alpha = 0.2
            
            if len(df['label'].unique().tolist()) == 1:
                ax.plot(x,y,'o',color=color,alpha=alpha)
                ax.set_xscale('log')
                ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
                texts.append(ax.text(x[-1], y[-1],str(round(cent,1))+' cm-1'))
                ax.set_xlim(x.min()/5,x.max()*5)
    
                if not conc_s:
                    my_xticks = toplot['label'].unique()
                    ax.set_xticks(toplot['Concentration [M]'].unique(), my_xticks)
                    plt.xlabel('arb. unit')
                else:
                    plt.xlabel('Concentration (M)')
                plt.ylabel('Intensity (a.u.)')
                plt.show()
                
            else:
                
                if y[-1] >= toplot['Intensity (a.u.)'].mean():
        
                    xnew = np.linspace(x.min(),x.max(),10000)
                    
                    if not conc_s:
                        fit = linear_fit
                        plt.xlabel('arb. unit')
                        ax.set_xscale('linear')
                        ax.set_xlim(x.min()-1,x.max()+1)
                        my_xticks = toplot['label'].unique()
                        plt.xticks(toplot['Concentration [M]'].unique(), my_xticks)
    #                    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
                    else:
                        fit = power_law
                        plt.xlabel('Concentration (M)')
                        ax.set_xscale('log')
                        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
                        ax.set_xlim(x.min()/5,x.max()*5)
                        
                    try:
                            
                        pars, cov = curve_fit(f=fit, xdata=x, ydata=y, p0=[1, 1])
                        stdevs = np.sqrt(np.diag(cov))
                        res = y - fit(x, *pars)
                        ss_res = np.sum(res**2)
                        ss_tot = np.sum((y-np.mean(y))**2)
                        r_squared = 1 - (ss_res / ss_tot)
                        
                        ax.plot(x, y,'o', color = color,alpha=alpha)
                        
#                        for sx,sy in zip(my_xticks,y):
#                            dil_peaks.append([cent,sy,sx])
                                
                        ax.plot(xnew, fit(xnew, *pars), linestyle='--', linewidth=2,color=color,alpha=alpha)
                        texts.append(ax.text(x[-1], y[-1],str(round(cent,1))+' cm-1'))
                        
                        if alpha == 1:
                            if not conc_s:
                                bestmatch = 'I(x)='+str(round(pars[0],2))+'(+/-'+str(round(stdevs[0],2))+')*x+'+str(round(pars[1],2))+'(+/-'+str(round(stdevs[1],2))+'))'
                                ax.text(x.min()*0.9, y.max()*0.9,bestmatch)
                                ax.text(x.min()*0.9, y.max()*0.85,'$R^{2}$ = '+str(round(r_squared,3))) 
                            else:
                                bestmatch = 'I(x)='+str(round(pars[0],2))+'(+/-'+str(round(stdevs[0],2))+')*x$^{'+str(round(pars[1],2))+'(+/-'+str(round(stdevs[1],2))+')}$'
                                ax.text(x.min()*0.9, y.max()*0.9,bestmatch)
                                ax.text(x.min()*0.9, y.max()*0.85,'$R^{2}$ = '+str(round(r_squared,3)))
                                
        
                            
                    except:
                        pass
                    
                plt.ylabel('Intensity (a.u.)')
                plt.show()
                
            adjust_text(texts, autoalign ='xy')  
    
#    pd.DataFrame(dil_peaks, columns= ['Raman shift (cm-1)','Intensity (a.u.)','Sample']).to_csv('dilution'+spectra.iloc[:,:-1].columns.tolist()[0]+'.csv',index=False)
        
            
    return bestmatch

###################################################################################################
    #loadinga analysis
    
def mapload(maps,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv,flag=1):
    f_dataset = []
    f_target = []
    f_mean = []
    for file in maps:
        target = []
        dataset = []
            
        wavenumber, intensity, n_points, n_raman,label,flag = ImportData(file)       
        if intensity.shape[1] == 0:
            flag = 0
        else:
            
            baseline = BaselineNAVG(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)
            norm = NormalizeNAVG(baseline,zscore,inv)
            f_mean.append(baseline.max().max())
            where_are_NaNs = np.isnan(norm)
            if sum(where_are_NaNs.values[0])>0:
                flag = 0
                norm[where_are_NaNs] = 0
            
            for i in range(norm.shape[1]):
                target.append(label)
                dataset.append(norm.iloc[:,i])
    #        if z == 0:
            for data,text in zip(dataset,target):
                f_dataset.append(data)
                f_target.append(text)
                flag = 1
                 
    return wavenumber,f_dataset, f_target,flag


def pcafit(norm, wavenumber, labels, windowmin = 0,  windowmax = 0):
    if windowmin == 0 and windowmin == 0:
        df_pca = pd.DataFrame(norm).reset_index(drop=True)
        target = pd.DataFrame(labels,columns=['sample'])
        
        pca = PCA()
        principalComponents = pca.fit_transform(df_pca)
        columns = ['principal component '+str(i+1) for i in range(principalComponents.shape[1])]
        principalDf = pd.DataFrame(data = principalComponents , columns = columns)
        df = pd.concat([principalDf, target], axis = 1)

    else:
        df_pca = pd.DataFrame(norm).reset_index(drop=True)
        target = pd.DataFrame(labels,columns=['sample'])
        
        region = (wavenumber > windowmin) & (wavenumber < windowmax)
        df_pca = df_pca.T[region].T
        wavenumber = wavenumber[region]
        
        pca = PCA()
        principalComponents = pca.fit_transform(df_pca)
        columns = ['principal component '+str(i+1) for i in range(principalComponents.shape[1])]

        principalDf = pd.DataFrame(data = principalComponents , columns = columns)
        df = pd.concat([principalDf, target], axis = 1)
   
    return pca,wavenumber,df

def plotload(pca,wavenumber,dim):
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_))
    if dim == 2:
        loadings = loadings.iloc[:,:2]
        loadings.columns = ['LPC1','LPC2']
    elif dim == 3:
        loadings = loadings.iloc[:,:3]
        loadings.columns = ['LPC1','LPC2','LPC3']

    return loadings

from itertools import chain


def plotprinc(df,loadings,wavenumber,pca,Q,smooth):
    
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
    
    colours = []
    
    for i in range(4):
        colours.append(cycle[i%len(cycle)])
    
    
    m1 = 1 
    m2 = 1
    if Q == 1: 
        m1 = 1 
        m2 = 1
        label = ['PC1P','PC2P']
    elif Q == 2:
        m1 = -1
        m2 = 1
        label = ['PC1N','PC2P']
    elif Q == 3:
        m2 = -1
        m1 = -1
        label = ['PC1N','PC2N']
    elif Q == 4:
        m1 = 1 
        m2 = -1
        label = ['PC1P','PC2N']
    else:
        m1 =[1,-1,-1,1]
        m2 = [1,1,-1,-1]
        label = [['PC1P','PC2P'],['PC1N','PC2P'],['PC1N','PC2N'],['PC1P','PC2N']]
    
    if Q != 'all':
        
        fig, axs = plt.subplots(4, 4,figsize=(12,12/1.618))
        gs = axs[1, 1].get_gridspec()  
    #    axs[2,2].remove()
        for ax in list(chain(*axs[2:,:])):
            ax.remove()
            
        for ax in list(chain(*axs[:2,2:])):
            ax.remove()
            
        for ax in list(chain(*axs[:2,:2])):
            ax.remove()
            

            
        axbig = fig.add_subplot(gs[2:,:])
        axbig2 = fig.add_subplot(gs[:2,2:])
        axbig3 = fig.add_subplot(gs[:2,:2])
        
        axbig2.plot(wavenumber,loadings['LPC1'],label = 'LPC1')
        axbig2.plot(wavenumber,loadings['LPC2'],label = 'LPC2')
        axbig2.legend(loc='best', frameon=False,prop={'size': 10})  
        
        axbig3.grid(True)
        axbig2.grid(True)
        
  
        NUM_COLORS = len(df['sample'].unique().tolist())
        
        cmaps = plt.get_cmap('gist_rainbow')
        
        colours2 = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
        

        for target,colo in zip(df['sample'].unique().tolist(),colours2):
            index = df['sample'] == target
            axbig3.plot(df['principal component 1'][index],df['principal component 2'][index],'o',label=target,c=colo)
        
        axbig3.fill_between(np.linspace(0, df['principal component 1'].max()*1.1,10),0,(df['principal component 2'].max()*1.1),alpha=0.2)
        axbig3.fill_between(np.linspace(df['principal component 1'].min()*1.1,0,10),0,(df['principal component 2'].max()*1.1),alpha=0.2)
        axbig3.fill_between(np.linspace(df['principal component 1'].min()*1.1,0,10),0,(df['principal component 2'].min()*1.1),alpha=0.2)
        axbig3.fill_between(np.linspace(0, df['principal component 1'].max()*1.1, 10),0,(df['principal component 2'].min()*1.1),alpha=0.2)

        
        axbig3.legend(loc='best', frameon=False,prop={'size': 8})
        axbig3.set_xlabel('PC1 : '+str(round(pca.explained_variance_ratio_[0]*100,2))+'%')
        axbig3.set_ylabel('PC2 : '+str(round(pca.explained_variance_ratio_[1]*100,2))+'%')
    
        axbig2.set_xlabel('wavenumber $(cm^{-1})$')
        axbig2.set_ylabel('Loadings')
        
        maxlod = loadings.max().max()
        
        axbig2.fill_between(wavenumber,maxlod*1.1,alpha=0.5,color='dimgrey')
        axbig2.fill_between(wavenumber,-maxlod*1.1,alpha=0.5,color='gainsboro')
        

        x = wavenumber
        y1 = smooth3(smooth*2,loadings['LPC1'])
        y2 = smooth3(smooth*2,loadings['LPC2'])
    
        peaksQ = []

        
        y1_m = y1*m1
        y2_m = y2*m2
        
        local_max_index2 = (y1_m>0) & (find_local_max(y1_m))
        stem2 = np.zeros(x.shape)
        stem2[local_max_index2] = 1 
    
#        local_max_index3 = s2(y2,0) & (find_local_max(y2))
        local_max_index3 = (y2_m>0) & (find_local_max(y2_m))
        stem3 = np.zeros(x.shape)
        stem3[local_max_index3] = 1 
        
#        plt.figure()
        axbig.plot(x[y1_m>0],y1_m[y1_m>0],'o')
        axbig.plot(x[y2_m>0],y2_m[y2_m>0],'o')
        axbig.plot(x[local_max_index2],y1_m[local_max_index2],'rx')
        axbig.plot(x[local_max_index3],y2_m[local_max_index3],'rx')
        axbig.text(x.min()*1.1,y1_m.max()*0.7,'Q'+str(Q),color=colours[Q-1])
        axbig.set_xlabel('Raman shift (cm$^{-1}$)')
        axbig.set_ylabel('Loadings (a.u.)')

        
        for wav in wavenumber[local_max_index2]:
            reg = fit_region(wavenumber,wav,10)
            
            if (y1_m[reg[reg==wav].index][0] > 0):
                try:
                    popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                                    reg, 
                                                                    y1_m[reg.index],
                                                                    p0=[y1_m[reg.index].max(), wav, 2/(np.pi*y1_m[reg.index].max())])
                    pars_1 = popt_1lorentz[0:3]
                    if (abs(pars_1[-1])>100) | (pars_1[0]<0):
                        pass
                    else:
                        axbig.plot(wavenumber,_1Lorentzian(wavenumber,*popt_1lorentz),lw=2)
                        axbig.fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),_1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
                        peaksQ.append([df['sample'][0]]+popt_1lorentz.tolist()+['Q'+str(Q)])

                except:
                    pass

                
        for wav in wavenumber[local_max_index3]:
            reg = fit_region(wavenumber,wav,10)
            
            if (y2_m[reg[reg==wav].index][0] > 0):
    
                try:
                    popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                                    reg, 
                                                                    y2_m[reg.index],
                                                                    p0=[y2_m[reg.index].max(), wav, 2/(np.pi*y2_m[reg.index].max())])
                    
                    pars_1 = popt_1lorentz[0:3]
                    if (abs(pars_1[-1])>100) | (pars_1[0]<0):
                        pass
                    else:
                        axbig.plot(wavenumber,_1Lorentzian(wavenumber,*popt_1lorentz),lw=2)
                        axbig.fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),
                                         _1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
                        
                        peaksQ.append([df['sample'][0]]+popt_1lorentz.tolist()+['Q'+str(Q)])
                
                except:
                    pass   

            peak = pd.DataFrame(peaksQ,columns=['label','height','center','width','importance'] )       
            
        peak.to_csv('loadingskey_'+'Q'+str(Q)+df['sample'][0]+'.csv',sep=';', index=False)

    else:
            
        fig, axs = plt.subplots(4, 4,figsize=(12,12/1.618))
        gs = axs[1, 1].get_gridspec()  
    #    axs[2,2].remove()
        for ax in list(chain(*axs[:2,:2])):
            ax.remove()
            
        for ax in list(chain(*axs[2:,:2])):
            ax.remove()
            
        for j in range(0,4):
            for ax in axs[j,2:]:
                ax.remove()
#            for ax in axs[2:,j]:
#                ax.remove()
    
            
        axbig = fig.add_subplot(gs[:2,:2])
        axbig2 = fig.add_subplot(gs[2:,:2])
        
        axsma = fig.add_subplot(gs[0,2:])
        axsma2 = fig.add_subplot(gs[1,2:])
        
        axsma3 = fig.add_subplot(gs[2,2:])
        axsma4 = fig.add_subplot(gs[3,2:])
        
        combax = [axsma,axsma2,axsma3,axsma4]
 
        axbig2.plot(wavenumber,loadings['LPC1'],label = 'LPC1')
        axbig2.plot(wavenumber,loadings['LPC2'],label = 'LPC2')
        axbig2.legend(loc='best', frameon=False,prop={'size': 10})  
        
        axbig.grid(True)
        axbig2.grid(True)
        
        NUM_COLORS = len(df['sample'].unique().tolist())
        
        cmaps = plt.get_cmap('gist_rainbow')
        
        colours2 = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
        

        for target,colo in zip(df['sample'].unique().tolist(),colours2):
            index = df['sample'] == target
            axbig.plot(df['principal component 1'][index],df['principal component 2'][index],'o',label=target,c=colo)
        
        axbig.fill_between(np.linspace(0, df['principal component 1'].max()*1.1, 10),0,(df['principal component 2'].max()*1.1),alpha=0.2)
        axbig.fill_between(np.linspace(df['principal component 1'].min()*1.1,0, 10),0,(df['principal component 2'].max()*1.1),alpha=0.2)
        axbig.fill_between(np.linspace(df['principal component 1'].min()*1.1,0, 10),0,(df['principal component 2'].min()*1.1),alpha=0.2)
        axbig.fill_between(np.linspace(0, df['principal component 1'].max()*1.1, 10),0,(df['principal component 2'].min()*1.1),alpha=0.2)
               
        
        axbig.legend(loc='best', frameon=False,prop={'size': 8})
        axbig.set_xlabel('PC1 : '+str(round(pca.explained_variance_ratio_[0]*100,2))+'%')
        axbig.set_ylabel('PC2 : '+str(round(pca.explained_variance_ratio_[1]*100,2))+'%')
    
        axbig2.set_xlabel('Raman shift (cm$^{-1}$)')
        axbig2.set_ylabel('Loadings')
        
        maxlod = loadings.max().max()
        
        axbig2.fill_between(wavenumber,maxlod*1.1,alpha=0.5,color='dimgrey')
        axbig2.fill_between(wavenumber,-maxlod*1.1,alpha=0.5,color='gainsboro')
        
        
    
        x = wavenumber
        y1 = smooth3(smooth*2,loadings['LPC1'])
        y2 = smooth3(smooth*2,loadings['LPC2'])
        
            
    
        i = 0 
    
        peaksQ = []
        
        for m_1,m_2 in zip(m1,m2):
            
    #        ax[i] = fig.add_subplot()
            
            y1_m = y1*m_1
            y2_m = y2*m_2
            
            local_max_index2 = (y1_m>0) & (find_local_max(y1_m))
            stem2 = np.zeros(x.shape)
            stem2[local_max_index2] = 1 
        
    #        local_max_index3 = s2(y2,0) & (find_local_max(y2))
            local_max_index3 = (y2_m>0) & (find_local_max(y2_m))
            stem3 = np.zeros(x.shape)
            stem3[local_max_index3] = 1 
            
    #        plt.figure()
            combax[i].plot(x[y1_m>0],y1_m[y1_m>0],'o')
            combax[i].plot(x[y2_m>0],y2_m[y2_m>0],'o')
            combax[i].plot(x[local_max_index2],y1_m[local_max_index2],'rx')
            combax[i].plot(x[local_max_index3],y2_m[local_max_index3],'rx')
            combax[i].text(x.min()*1.1,y1_m.max()*0.7,'Q'+str(i+1),color=colours[i])
    
            combax[i].set_xlabel('Raman shift (cm$^{-1}$)')
            combax[i].set_ylabel('L(a.u.)')
            
    
            
            for wav in wavenumber[local_max_index2]:
                reg = fit_region(wavenumber,wav,10)
                
                if (y1_m[reg[reg==wav].index][0] > 0):
                    try:
                        popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                                        reg, 
                                                                        y1_m[reg.index],
                                                                        p0=[y1_m[reg.index].max(), wav, 2/(np.pi*y1_m[reg.index].max())])
                        pars_1 = popt_1lorentz[0:3]
                        
                        if (abs(pars_1[-1])>100) | (pars_1[0]<0):
                            pass
                        else:
                            
                            combax[i].plot(wavenumber,_1Lorentzian(wavenumber,*popt_1lorentz),lw=2)
                            combax[i].fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),_1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
                            peaksQ.append([df['sample'][0]]+popt_1lorentz.tolist()+['Q'+str(i+1)])
        
                    except:
                        pass
    
                    
            for wav in wavenumber[local_max_index3]:
                reg = fit_region(wavenumber,wav,10)
                
                if (y2_m[reg[reg==wav].index][0] > 0):
        
                    try:
                        popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                                        reg, 
                                                                        y2_m[reg.index],
                                                                        p0=[y2_m[reg.index].max(), wav, 2/(np.pi*y2_m[reg.index].max())])
                        
           
                        pars_1 = popt_1lorentz[0:3]
                        if (abs(pars_1[-1])>100) | (pars_1[0]<0):
                            pass
                        else:
    
                            combax[i].plot(wavenumber,_1Lorentzian(wavenumber,*popt_1lorentz),lw=2)
                            combax[i].fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),
                                             _1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
    
                            peaksQ.append([df['sample'][0]]+popt_1lorentz.tolist()+['Q'+str(i+1)])
                        
                    except:
                        pass   
                    
            i = i + 1
            
            peak = pd.DataFrame(peaksQ,columns=['label','height','center','width','importance'] )       
             
        peak.to_csv('loadings'+Q+df['sample'][0]+'.csv',sep=';', index=False)
    
    
    
    plt.legend(loc='best', frameon=False,prop={'size': 8})    
    fig.tight_layout()
#    plt.grid(True)
    plt.show()
   


#maps = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\all saliva beguining\saliva_s.txt',
#        r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\all saliva beguining\saliva_ms.txt']
#  
#smooth=7
#asymmetry_param = 0.05
#smoothness_param = 1000000
#max_iters = 10
#conv_thresh =0.00001
#zscore=0
#inv=0  
#Q=1
#windowmin=0
#windowmax=0
#dim = 2
#smooth = 7

def loadingpeak(maps,smooth,
                asymmetry_param, 
                smoothness_param, 
                max_iters, conv_thresh,
                zscore,inv,  Q, windowmin,
                windowmax, dim=2,flag=1):
    
    wavenumber,norm,labels,flag =  mapload(maps,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv,flag=1)
    pca,wavenumber,df = pcafit(norm, wavenumber,labels, windowmin,  windowmax)
    loadings = plotload(pca,wavenumber,dim)
#    loadings = loadings-loadings.median(axis=0)
    plotprinc(df,loadings,wavenumber,pca,Q,smooth)
    



#instead of ploting the PC1 and PC2 finding which of the PCi PCj has more distance between them
#calculate the cluster difference matrix, and them plot the max distance
#sort distances from higher to lower
#get the PCs where that happens
#pass them onto the loadings peak feating
    
from scipy import stats

#Q = 'all'
#smooth = 7
#dist_n = 5

def pca_max_dist(maps,smooth,
                asymmetry_param, 
                smoothness_param, 
                max_iters, conv_thresh,
                zscore,inv,  Q, windowmin,
                windowmax, dist_n,dim=2,flag=1):
#(df,pca,wavenumber,Q,smooth,dist_n):
    
    wavenumber,norm,labels,flag =  mapload(maps,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv,flag=1)
    pca,wavenumber,df = pcafit(norm, wavenumber,labels, windowmin,  windowmax)

    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_),
                                columns=['principal component '+ str(m+1) for m in range(pca.components_.T.shape[1])])
        

    position = []

    
    labels = df['sample']
    PCs =  df.iloc[:,:-1]
    
#    iterval = np.argmax(np.cumsum(pca.explained_variance_ratio_)>0.95)
    iterval = 10
        
    for label in list(np.unique(labels)):
        for i in range(iterval):
            for j in range(iterval):
                if j>i:
                        
                    index = label == labels
                    
                    gmm = GaussianMixture(n_components=1).fit(pd.concat([PCs.loc[index].iloc[:,i] , 
                                         PCs.loc[index].iloc[:,j]],axis=1).values)
    
                    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):

                        if covar.shape == (2, 2):
                            U, s, Vt = np.linalg.svd(covar)
                            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
                            width, height = 2 * np.sqrt(s)
                        else:
                            angle = 0
                            width, height = 2 * np.sqrt(covar)

                    position.append([pos,width,height,label,PCs.loc[index].iloc[:,i].name,PCs.loc[index].iloc[:,j].name])
                    
                else:
                    pass
            
           
    dist_matrix = []               

    param = 2
    sampl = (iterval-1)*iterval//param
    
    #both PCi and PCj needs to be the same for the different 
    
    for i in range(sampl):                   
                
        x1 = position[i][0][0]
        x2 = position[i+sampl][0][0]
        y1 = position[i][0][1]
        y2 = position[i+sampl][0][1]
        r = np.sqrt((x1-x2)**2+(y1-y2)**2) 
        
        w1 =  position[i][1]
        w2 = position[i+sampl][1]
        h1 = position[i][2]
        h2 = position[i+sampl][2]
        er = abs(np.sqrt((w1/2)**2+(h1/2)**2)-np.sqrt((w2/2)**2+(h2/2)**2))
#                    er = np.sqrt(((x1-x2)*w1)**2+((y1-y2)*h1)**2+((x2-x1)*w2)**2+((y2-y1)*h2)**2)/r
        
        dist_matrix.append([r , er ,  position[i][3], position[i][4],position[i][5],position[i+sampl][3]])
    else:
        pass

    
    dist = pd.DataFrame(dist_matrix,columns=['dist','er','label0','PCi','PCf','label1'])
#    dist['prod'] = dist['dist']/dist['er']
    dist = dist.sort_values(by='dist',ascending=False).reset_index(drop=True)
      
    
    text = []
    
    plt.rcParams['figure.autolayout'] = True
    
    peaksQ = []
    
    for k in range(dist_n):
        lab1 = dist['label0'].iloc[k]
        lab2 = dist['label1'].iloc[k]
        PC = [dist.iloc[k]['PCi'] , dist.iloc[k]['PCf']]
      
        x1 = df[df['sample']==lab1][PC[0]]
        y1 = df[df['sample']==lab1][PC[1]]
        
        x2 = df[df['sample']==lab2][PC[0]]
        y2 = df[df['sample']==lab2][PC[1]]
        
#        print(PC[0])
#        print(PC[1])
        
        pclab1 = 'PC ' + PC[0].split()[-1]
        pclab2 = 'PC ' + PC[1].split()[-1]
        
        z1 = (np.abs(stats.zscore(x1))<3) & (np.abs(stats.zscore(y1))<3)
        z2 = (np.abs(stats.zscore(x2))<3) & (np.abs(stats.zscore(y2))<3)
            
        data1 = [x1[z1],y1[z1]]
        data2 = [x2[z2],y2[z2]]

#        data1 = [x1 , y1]
#        data2 = [x2 , y2]

       
#        loadings = loadings-loadings.median(axis=0)
        
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
        colours = [] 
        
        for n in range(4):
            colours.append(cycle[n%len(cycle)])
        
        
        m1 = 1 
        m2 = 1
        if Q == 1: 
            m1 = 1 
            m2 = 1
            label = ['PCP','PCP']
        elif Q == 2: 
            m1 = -1
            m2 = 1
            label = ['PCN','PCP']
        elif Q == 3:
            m2 = -1
            m1 = -1
            label = ['PCN','PCN']
        elif Q == 4:
            m1 = 1 
            m2 = -1
            label = ['PCP','PCN']
        else:
            m1 =[1,-1,-1,1]
            m2 = [1,1,-1,-1]
            label = [['PCP','PCP'],['PCN','PCP'],['PCN','PCN'],['PCP','PCN']]
        
        fig, axs = plt.subplots(4, 4,figsize=(12,12/1.618))
        gs = axs[1, 1].get_gridspec()  
    #    axs[2,2].remove()
        for ax in list(chain(*axs[:2,:2])):
            ax.remove()
            
        for ax in list(chain(*axs[2:,:2])):
            ax.remove()
            
        for j in range(0,4):
            for ax in axs[j,2:]:
                ax.remove()
#            for ax in axs[2:,j]:
#                ax.remove()
    
            
        axbig = fig.add_subplot(gs[:2,:2])
        axbig2 = fig.add_subplot(gs[2:,:2])
        
        axsma = fig.add_subplot(gs[0,2:])
        axsma2 = fig.add_subplot(gs[1,2:])
        
        axsma3 = fig.add_subplot(gs[2,2:])
        axsma4 = fig.add_subplot(gs[3,2:])
        
        combax = [axsma,axsma2,axsma3,axsma4]
 
        axbig2.plot(wavenumber,loadings[PC[0]],label = pclab1)
        axbig2.plot(wavenumber,loadings[PC[1]],label = pclab2)
        axbig2.legend(loc='best', frameon=False,prop={'size': 10})  
        
        axbig.grid(True)
        axbig2.grid(True)
        
        
        axbig.plot(data1[0],data1[1],'o',label=lab1)
        axbig.plot(data2[0],data2[1],'o',label=lab2)
        
#        data0 = [pd.concat([data1[0],data2[0]]).reset_index(drop=True),
#                 pd.concat([data1[1],data2[1]]).reset_index(drop=True)]
        
        data0 = [pd.concat([data2[0],data1[0]]), pd.concat([data2[1],data1[1]])]
        
        axbig.fill_between(np.linspace(0, data0[0].max()*1.1,10),0,(data0[1].max()*1.1),alpha=0.2)
        axbig.fill_between(np.linspace(data0[0].min()*1.1,0,10),0,(data0[1].max()*1.1),alpha=0.2)
        axbig.fill_between(np.linspace(data0[0].min()*1.1,0,10),0,(data0[1].min()*1.1),alpha=0.2)
        axbig.fill_between(np.linspace(0, data0[0].max()*1.1, 10),0,(data0[1].min()*1.1),alpha=0.2)
        axbig.legend(loc='best', frameon=False,prop={'size': 8})
        axbig.set_xlabel(pclab1+' '+str(round(pca.explained_variance_ratio_[loadings.columns==PC[0]][0]*100,2))+'%')
        axbig.set_ylabel(pclab2+' '+str(round(pca.explained_variance_ratio_[loadings.columns==PC[1]][0]*100,2))+'%')
    
        axbig2.set_xlabel('wavenumber $(cm^{-1})$')
        axbig2.set_ylabel('Loadings')
                
        
        maxlod = np.maximum(np.abs(loadings[PC[0]]).max(),np.abs(loadings[PC[1]]).max())
        
        axbig2.fill_between(wavenumber,maxlod*1.1,alpha=0.5,color='dimgrey')
        axbig2.fill_between(wavenumber,-maxlod*1.1,alpha=0.5,color='gainsboro')
        

        x = wavenumber
        y1 = smooth3(smooth*2,loadings[PC[0]])
        y2 = smooth3(smooth*2,loadings[PC[1]])
        



        i = 0 
    
        
        
        for m_1,m_2 in zip(m1,m2):
            
    #        ax[i] = fig.add_subplot()
            
            y1_m = y1*m_1
            y2_m = y2*m_2
            
            local_max_index2 = (y1_m>0) & (find_local_max(y1_m))
            stem2 = np.zeros(x.shape)
            stem2[local_max_index2] = 1 
        
    #        local_max_index3 = s2(y2,0) & (find_local_max(y2))
            local_max_index3 = (y2_m>0) & (find_local_max(y2_m))
            stem3 = np.zeros(x.shape)
            stem3[local_max_index3] = 1 
            
    #        plt.figure()
            combax[i].plot(x[y1_m>0],y1_m[y1_m>0],'o')
            combax[i].plot(x[y2_m>0],y2_m[y2_m>0],'o')
            combax[i].plot(x[local_max_index2],y1_m[local_max_index2],'rx')
            combax[i].plot(x[local_max_index3],y2_m[local_max_index3],'rx')
            
            
            combax[i].text(x.min()*1.1,y1_m.max()*0.7,'Q'+str(i+1),color=colours[i])
    
            combax[i].set_xlabel('Raman shift (cm$^{-1}$)')
            combax[i].set_ylabel('L(a.u.)')
            
#            try:
#                combax[i].set_ylim(ymin = np.min([y1_m[local_max_index2],y2_m[local_max_index2]])*1.1,
#                  ymax = np.max([y1_m[local_max_index2],y2_m[local_max_index2]])*1.1)
#            except:
#                pass
#            
    
            
            for wav in wavenumber[local_max_index2]:
                reg = fit_region(wavenumber,wav,10)
                
                if (y1_m[reg[reg==wav].index][0] > 0):
                    try:
                        popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                                        reg, 
                                                                        y1_m[reg.index],
                                                                        p0=[y1_m[reg.index].max(), wav, 2/(np.pi*y1_m[reg.index].max())])
                        pars_1 = popt_1lorentz[0:3]
                        
                        if (abs(pars_1[-1])>100) | (pars_1[0]<0):
                            pass
                        else:
                            combax[i].plot(wavenumber,_1Lorentzian(wavenumber,*popt_1lorentz),lw=2)
                            combax[i].fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),_1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
                            peaksQ.append([df['sample'][0]]+popt_1lorentz.tolist()+['Q'+str(i+1)]+[pclab1+pclab2])
#                            print([df['sample'][0]]+popt_1lorentz.tolist()+['Q'+str(i+1)])

    
                    except:
                        pass
    
                    
            for wav in wavenumber[local_max_index3]:
                reg = fit_region(wavenumber,wav,10) 
                
                if (y2_m[reg[reg==wav].index][0] > 0):
        
                    try:
                        popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                                        reg, 
                                                                        y2_m[reg.index],
                                                                        p0=[y2_m[reg.index].max(), wav, 2/(np.pi*y2_m[reg.index].max())])
                        
           
                        pars_1 = popt_1lorentz[0:3]                       
                        if (abs(pars_1[-1])>100) | (pars_1[0]<0):
                            pass
                        else:
                            combax[i].plot(wavenumber,_1Lorentzian(wavenumber,*popt_1lorentz),lw=2)
                            combax[i].fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),
                                             _1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
     
                            peaksQ.append([df['sample'][0]]+popt_1lorentz.tolist()+['Q'+str(i+1)]+[pclab1+pclab2])
#                            print([df['sample'][0]]+popt_1lorentz.tolist()+['Q'+str(i+1)])


                        
                    except:
                        pass   
                    
            i = i + 1
        


        axbig.legend(loc='best', frameon=False,prop={'size': 8})    
        axbig2.legend(loc='best', frameon=False,prop={'size': 8})  
        fig.subplots_adjust(hspace=1, wspace=1) 
        fig.tight_layout()
    #    plt.grid(True)
        plt.show()
        
        
        text.append([str(round(dist.iloc[k]['dist'],3))+' +/- '+
                     str(round(dist.iloc[k]['er'],3))+' '+pclab1+' - '+pclab2])
        
    
    peak = pd.DataFrame(peaksQ,columns=['label','height','center','width','importance','PCs'] )       
             
    peak.to_csv('loadings'+Q+df['sample'][0]+'.csv',sep=';', index=False)
    
    
    return text



def cluster(data, maxgap):
    data.sort()
    groups = [[data[0]]]
    for item in data[1:]:
        val = abs(item - groups[-1][-1])
        if val <= maxgap:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups    


##########################################################################################################
    ############################PEAK FINDER ALL IN ONE
    #########################################################################################################
#
   
#spectra = sp
#spectra_std = pd.DataFrame()
#const = sp.std().mean()
    
def diff2(spectra,spectra_std,const):
    
    new_spec = []
    for i in range(spectra.shape[1]):
         
        if spectra_std.empty:     
            if i == 0:
                minim = spectra.iloc[:,i].min()
                z = spectra.iloc[:,i]-minim
                new_spec.append(z)
                minim = z
            else:
                minim = (spectra.iloc[:,i]-minim).min()
                z = spectra.iloc[:,i]-minim+const
                new_spec.append(z)
                minim = z 
    
        else:
            
            zero = pd.DataFrame([0]*spectra_std.shape[0])
            spectra_std = pd.concat([spectra_std,zero],axis=1)


            if i == 0:
                minim = spectra.iloc[:,i].min()
                z = spectra.iloc[:,i]-minim+spectra_std.iloc[:,i].max()
                new_spec.append(z)
                minim = z

            else:
                minim = (spectra.iloc[:,i]-minim).min()
                z = spectra.iloc[:,i]-minim+spectra_std.iloc[:,i].max()+abs((spectra_std.iloc[:,i-1]+spectra_std.iloc[:,i])).min()+const
                new_spec.append(z)
                minim = z   

                
    return pd.DataFrame(new_spec).T 
    
#spectra = pd.DataFrame(new_spec).T 
    
#lockgens = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\saliva_all.txt',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\milk_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt']
#
#keygens = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keysaliva_all..csv']
#
#tol = 1
#
#operator = 'standard'



def fit_region2(wavenumber,pos,tol = 10):
    index = []
    if type(pos) is not list:
        pos = [pos]
    for p in pos:
        min_l = min(wavenumber.index.tolist())
        max_l = max(wavenumber.index.tolist())
        index.append(wavenumber.index[(wavenumber>p-1) & (wavenumber<p+1)].tolist()[0])
    indexm = np.min(index)
    indexM = np.max(index)
    index1 = indexm-tol
    index2 = indexM+tol
    if index1<min_l:
        index1 = 0
    if index2>max_l:
        index2 = max_l
        
    return wavenumber[index1:index2]


def _nLorentzian(x,*arg):
    i = len(arg)//3
    amp = arg[0*i:i*1]
    cent = arg[i*1:i*2]
    wid = arg[i*2:i*3]
    return np.sum([((a*b**2)/((x-c)**2+b**2)+(a*b*np.exp(-0.5*((x-c)/b)**2))) for a,b,c in zip(amp,wid,cent)],axis=0)
                           
            



def deconvoluted_Lorentzian(wavenumber,intensity_b,wav):
    reg = fit_region2(wavenumber,wav,15)
    data = []
    amp = []
    cent = []
    wid = []
    
    xx = np.linspace(reg.min(),reg.max(),1000)
    spl = make_interp_spline(sorted(reg),intensity_b[reg.index][::-1])
    yy = spl(xx)

    
    for w in wav:
        
        amp.append(np.mean(intensity_b[(wavenumber<w+1)&(wavenumber>w-1)]))
        cent.append(w)
        wid.append(np.mean(2/(np.pi*intensity_b[(wavenumber<w+1)&(wavenumber>w-1)])))
        
    try:
        data.append(scipy.optimize.curve_fit(_nLorentzian, 
                                 xx, 
                                 yy,
                                 p0=[amp,cent,wid],method='lm',
                                     maxfev=20000))
    except:
        pass
    
    return data
    

    


    
def peakmatching(keygens,lockgens,tol,operator):
    if operator == 'standard':
        
        all_spec = []
        dataset = []  
        for lockgen in lockgens:
                
            if lockgen.split('\\')[-1].split('.')[-1] == 'txt':
                
                peak, spectra = peakfinder([lockgen],smooth=7,asymmetry_param = 0.05,smoothness_param = 1000000,max_iters = 10,conv_thresh =0.00001)                
                peak = peak[peak['Intensity (a.u.)']>0]
                lock,conc_s = clean_up_rep(peak)
                all_spec.append(spectra)

            else:
                lock = pd.read_csv(lockgen,sep=';')
                lock = lock[lock['Intensity (a.u.)']>0]
                
                #lorentzian fit to the major important peaks
                
                xnew = np.linspace(lock['center'].min(),lock['center'].max(),1000)
                xnew = xnew[(xnew>100) & (xnew<3200)]
                
                aux_spec = []
                for cent,inten,wid in zip(lock['center'].tolist(),lock['Intensity (a.u.)'].tolist(),lock['width'].tolist()):
                    aux_spec.append(_1Lorentzian(xnew,inten,cent,wid))
                    
                aux_spec = pd.DataFrame(np.sum(aux_spec,axis=0))
                aux_spec.columns = lock['label'].unique().tolist()
                aux_spec['wavenumber'] = xnew
                  
                all_spec.append(aux_spec)

            for keygen in keygens:
                
                key = pd.read_csv(keygen,sep=';')
                key = key[key['Intensity (a.u.)']>0]
                
                if tol>1:
                    lock = lock[lock['importance']=='major']
                    key = key[key['importance']=='major']
    

            
                for k in key['center'].unique():
                    for l in lock['center'].unique():
                        if (l<k+tol) & (k-tol<l):
                            dataset.append([k,key['label'][key['center'] == k].values[0],lock['label'][lock['center'] == l].values[0]])
                            

        spec = []
        for spe in all_spec:
            spec.append(spe.iloc[:,0])
            wavenumber = spe.iloc[:,1]
        
        spec = pd.DataFrame(spec).T

        stack = diff2(spec,pd.DataFrame(),spec.max().min())
        
        fig,ax = plt.subplots(figsize=(9,9/1.618)) 
        ax.plot(wavenumber,stack, label = stack.columns.tolist())  
        
        for data in dataset:
            ax.fill_between(np.linspace(data[0]-tol,data[0]+tol,10),stack[data[-1]].max(),stack[data[-1]].min(),color='yellow',alpha=1/3)
            ax.text(data[0]-tol,stack[data[-1]].min(),data[1][:4],rotation=90,fontsize='xx-small')
            
        ax.legend(loc='best',frameon=False)
        ax.set_xlabel('Raman shift $(cm^{-1})$')
        ax.set_ylabel('Intensity (a. u.)')
        plt.show()
        
        
            
        df_dataset = pd.DataFrame(dataset,columns=['center','key','lock'])
        
        index = (df_dataset['center']>wavenumber.min()) & (df_dataset['center']<wavenumber.max())
        df_dataset = df_dataset[index].reset_index(drop=True)
        
        
        max_value = []
        min_value = []
        
        lab = stack.columns.tolist()
        fig2,ax2 = plt.subplots(figsize=(9,9/1.618))
        ax2.plot(wavenumber,stack, label = lab)  
        fig3,ax3 = plt.subplots(figsize=(9,9/1.618))
        
        for i in range(stack.shape[1]):

            intensity_b = stack.iloc[:,i]-stack.iloc[:,i].min()
            max_value.append(intensity_b.max())
            min_value.append(intensity_b.min())

            
            smoothed_signal = smooth2(7,  intensity_b)
            smoothed_signal = Normalize(smoothed_signal,0,0)
             
            lorentz = []
            
            
            clean_wav = cluster(sorted(df_dataset[df_dataset['lock']==lab[i]]['center'].values),10)

            NUM_COLORS = len(clean_wav)
    
            cmaps = plt.get_cmap('gist_rainbow')
            
            colours = [cmaps(1.*k/NUM_COLORS) for k in range(NUM_COLORS)]

            for count,wav in enumerate(clean_wav):
                


                data = deconvoluted_Lorentzian(wavenumber,intensity_b,wav)
                

                
                if  data == [] or any(data[0][0][:len(wav)]>intensity_b.max()*1.2):
                    continue
                 

                    
                for pop,pcov in data:
                    ax2.plot(wavenumber,_nLorentzian(wavenumber,*pop)+stack.iloc[:,i].min())
                    ax2.fill_between(wavenumber, _nLorentzian(wavenumber,*pop)+stack.iloc[:,i].min(),
                                         stack.iloc[:,i].min(), color = colours[count],
                                         label= df_dataset['key'].unique()[i][:4]+' at '+str([round(a,1) for a in wav]),alpha=0.5)
                    
                    ax2.legend(loc='best', fontsize='xx-small',frameon=False,ncol=4)
                    ax2.set_xlabel('Raman shift $(cm^{-1})$')
                    ax2.set_ylabel('Intensity (a. u.)')
        
                    lorentz.append(_nLorentzian(wavenumber,*pop))
                    
                    
            lorentz_all =pd.DataFrame(lorentz).sum() 
            residual_1lorentz = intensity_b - lorentz_all
            
    
            ax3.plot(wavenumber,residual_1lorentz+min_value[i], label = lab[i][:4]+' - '+df_dataset['key'].unique()[i][:4]+' residue')  
            ax3.hlines(0,wavenumber.min(),wavenumber.max(),ls='--',color='r')
            ax3.legend(loc='best', prop={'size':12},frameon=False)
            ax3.set_xlabel('Raman shift $(cm^{-1})$')
            ax3.set_ylabel('Intensity (a. u.)')
                    
            

        
    if operator == 'loading':
            
        colours = []
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
        dataset = []     
        
        for i in range(4):
            colours.append(cycle[i%len(cycle)])
    
        for keygen in keygens:
            
            key = pd.read_csv(keygen,sep=';')     
            key = key[key['Intensity (a.u.)']>0]
    
            lock = pd.read_csv(lockgens[0], sep=';')
            lock = lock[lock['height']>0]
            lock = lock[(lock['center']>key['center'].min()) & (lock['center']<key['center'].max())]
                   
             
            for k in key['center'].unique():
                for l in lock['center'].unique():
                    if (l<k+tol) & (k-tol<l):
                        dataset.append([k,key['label'][key['center'] == k].values[0],lock['importance'][lock['center'] == l].tolist()])
    
        sim_plot_sep = []
        xnew2 = np.linspace(lock['center'].min(),lock['center'].max(),1000)
        for imp in lock['importance'].unique().tolist():
            aux = []
            index = lock['importance'] == imp
            for cent,inten,wid in zip(lock['center'][index].tolist(),lock['height'][index].tolist(),lock['width'][index].tolist()):
                aux.append(_1Lorentzian(xnew2,inten,cent,wid))
            sim_plot_sep.append(pd.DataFrame(aux).sum().values)
        
        sim_plot_sep = pd.DataFrame(sim_plot_sep)
        sim_plot_sep['importance'] = pd.DataFrame(lock['importance'].unique().tolist()) 
    
        Qs = sim_plot_sep['importance']
        
        order = []
        for imp,color in zip(Qs.tolist(),colours):
            order.append([imp,sim_plot_sep.iloc[:,:-1][Qs==imp].values[0].sum(),color])
        order = pd.DataFrame(order,columns=['Qs','sum','color']).sort_values(by='sum')
    
        stack = []
        for imp in order['Qs'].tolist():
            y = sim_plot_sep.iloc[:,:-1][Qs==imp].values[0]
            yp = y-als_baseline(y)
            stack.append(yp) 
    #        plt.plot(xnew,yp)
            
        
        stack_df = pd.DataFrame(stack).T
        stack_df.columns = order['Qs'].tolist()
        spectra = diff2(stack_df,pd.DataFrame(),stack_df.std().mean())
        plt.figure(figsize=(9,9/1.618))
        for Qs in spectra.columns.tolist():
            plt.plot(xnew2,spectra[Qs], label = Qs,lw=2,color=order['color'][order['Qs']==Qs].values[0])
    
        for data in dataset:
#            plt.text(data[0]-tol,0.8*spectra.max().max(),data[1],rotation=90,fontsize='xx-small')
    #            plt.text(data[0]-tol,spectra.max().max(),(", ".join(data[2])),fontsize='xx-small')
            for Qs in spectra.columns.tolist():  
                for data2 in data[2]:
                    if Qs == data2:
                        plt.fill_between(np.linspace(data[0]-tol,data[0]+tol,10),spectra[Qs].max(),spectra[Qs].min(),color='yellow',alpha=1/3)
                        plt.text(data[0]-tol,spectra[Qs].min(),data[1],rotation=90,fontsize='xx-small')
                
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel('Loadings (a.u.)')
        labels = np.arange(xnew2.min()-xnew2.min()%50,xnew2.max()-xnew2.max()%50+51,50)
        plt.xticks(labels,rotation=90)
        plt.legend(loc='best', prop={'size':12},frameon=False)
    #    plt.yticks([])
        plt.show()
                
    if operator == 'som':                

        dataset = []         
        for keygen in keygens:
            
            key = pd.read_csv(keygen,sep=';')     
            key = key[key['Intensity (a.u.)']>0]
    
            lock = pd.read_csv(lockgens[0], sep=';')
            lock = lock[lock['height']>0]
            lock = lock[(lock['center']>key['center'].min()) & (lock['center']<key['center'].max())]
            
            
            NUM_COLORS = len(lock['importance'].unique())
            
            cmaps = plt.get_cmap('gist_rainbow')
            
            colours = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
            
#            colours = []
#            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
#           
#            for i in range(len(lock['importance'].unique())):
#                colours.append(cycle[i%len(cycle)])
                   
             
            for k in key['center'].unique():
                for l in lock['center'].unique():
                    if (l<k+tol) & (k-tol<l):
                        dataset.append([k,key['label'][key['center'] == k].values[0],lock['importance'][lock['center'] == l].tolist()])
    
        sim_plot_sep = []
        xnew2 = np.linspace(lock['center'].min(),lock['center'].max(),1000)
        for imp in lock['importance'].unique().tolist():
            aux = []
            index = lock['importance'] == imp
            for cent,inten,wid in zip(lock['center'][index].tolist(),lock['height'][index].tolist(),lock['width'][index].tolist()):
                aux.append(_1Lorentzian(xnew2,inten,cent,wid))
            sim_plot_sep.append(pd.DataFrame(aux).sum().values)
        
        sim_plot_sep = pd.DataFrame(sim_plot_sep)
        sim_plot_sep['importance'] = pd.DataFrame(lock['importance'].unique().tolist()) 
    
        Qs = sim_plot_sep['importance']
        
        order = []
        for imp,color in zip(Qs.tolist(),colours):
            order.append([imp,sim_plot_sep.iloc[:,:-1][Qs==imp].values[0].sum(),color])
        order = pd.DataFrame(order,columns=['Qs','sum','color']).sort_values(by='sum')
    
        stack = []
        for imp in order['Qs'].tolist():
            y = sim_plot_sep.iloc[:,:-1][Qs==imp].values[0]
            yp = y-als_baseline(y)
            stack.append(yp) 
    #        plt.plot(xnew,yp)
            
        
        stack_df = pd.DataFrame(stack).T
        stack_df.columns = order['Qs'].tolist()
        spectra = diff2(stack_df,pd.DataFrame(),stack_df.std().mean())
        plt.figure(figsize=(9,9/1.618))
        for Qs in spectra.columns.tolist():
            plt.plot(xnew2,spectra[Qs], label = Qs,lw=2,color=order['color'][order['Qs']==Qs].values[0])
    
        for data in dataset:
#            plt.text(data[0]-tol,0.8*spectra.max().max(),data[1],rotation=90,fontsize='xx-small')
    #            plt.text(data[0]-tol,spectra.max().max(),(", ".join(data[2])),fontsize='xx-small')
            for Qs in spectra.columns.tolist():  
                for data2 in data[2]:
                    if Qs == data2:
                        plt.fill_between(np.linspace(data[0]-tol,data[0]+tol,10),spectra[Qs].max(),spectra[Qs].min(),color='yellow',alpha=1/3)
                        plt.text(data[0]-tol,spectra[Qs].min(),data[1],rotation=90,fontsize='xx-small')
                
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel('Loadings (a.u.)')
        labels = np.arange(xnew2.min()-xnew2.min()%50,xnew2.max()-xnew2.max()%50+51,50)
        plt.xticks(labels,rotation=90)
        plt.legend(loc='best', prop={'size':12},frameon=False)
    #    plt.yticks([])
        plt.show()


##################SOM##########################
###############################################
#windowmin=700
#windowmax=1700
##inputFilePath=files
#smooth=7
#asymmetry_param = 0.05
#smoothness_param = 1000000
#max_iters = 10
#conv_thresh =0.00001
#zscore=3
#inv=0  
#    
##keygen = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\01st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
##          r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\02nd_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
##          r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\03rd_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
##          r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\04th_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
##          r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\05th_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt']
#neuron = 10
#
#
#keygen = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\emma\Male vs Female\Female.txt',
#          r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\emma\Male vs Female\Male.txt']
#
#
#data = ImportDataSOM(keygen,windowmin,windowmax,zscore)
#som = LoadSOM(data,neuron)
#PlotAvgSpec(data,windowmin,windowmax)
#HexagonaSOMPlot(som,data,neuron)
#NeuronActivationperWavePlot(som, data)
#text = Classification(som,data)
#NeuronActivationperPlot(som, data,smooth,neuron)

import matplotlib.patches as patches

#keygens = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keysaliva_785n_milk_785nm_.csv'
#
#lockgens = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\saliva peaks from the literature.csv'

def NeuronActivationperPlot(som, data,smooth,neuron):   
    label = data.index.drop_duplicates().values.tolist()
    label =  '_'.join(label[5:])
    
    
    dataval = data.values
    columns = list(data.columns)
    win_map = som.win_map(dataval)
    size=som.distance_map().shape[0]
    
    #properties columns 0 and 1 are related to the positions, the rest are related to the weights
    #+2 to add space to insert the row and col place in the hexagonalbins
    
    properties = np.empty((size*size,2+dataval.shape[1]))
    properties[:]=np.NaN
    
    
    for row in range(0,size):
        for col in range(0,size):
            properties[size*row+col,0]=row
            properties[size*row+col,1]=col
    
    for position, values in win_map.items():
        properties[size*position[0]+position[1],0]=position[0]
        properties[size*position[0]+position[1],1]=position[1]
        properties[size*position[0]+position[1],2:] = np.mean(values, axis=0)
    
    
    B = ['row', 'col']
    B.extend(columns)
    properties = pd.DataFrame(data=properties, columns=B)
    
    fig2,axes = plt.subplots(neuron,neuron,sharex=True,sharey=True)
    fig2.subplots_adjust(hspace=0, wspace=0)
    
    peak_vector = []
    max_value = []
    spectra = []
    err = []
    
    for i in range(properties.shape[0]):
                
        if math.isnan(properties.iloc[i,2:].values[0]):
            pass
        else:
            
            r = properties['row'].loc[i]
            c = properties['col'].loc[i]
            
            fig,ax = plt.subplots(figsize=(9,9/1.618))
            ax.plot(properties.iloc[i,2:].index.values ,properties.iloc[i,2:].values)
            axes[abs(int(c)-neuron+1),int(r)].plot(properties.iloc[i,2:].index.values ,properties.iloc[i,2:].values)
            # Create a Rectangle patch
            
            rangex = properties.iloc[i,2:].index.max()-properties.iloc[i,2:].index.min()
            rangey = properties.iloc[:,2:].max().max() - properties.iloc[:,2:].min().min()
            
            patchx = properties.iloc[i,2:].index.max()*0.95
            patchy = rangey/neuron
            l = rangex//3/1.618
            w = rangey//3
            
            
            rect = patches.Rectangle((patchx, patchy) , l , w, edgecolor='k', facecolor='none')
            
            k = 1
            while(l>l*k/neuron):
        
                rect2 = patches.Rectangle((patchx+k*l/neuron, patchy) , l/neuron , w, edgecolor='k', facecolor='none')
                rect3 = patches.Rectangle((patchx, patchy+k*w/neuron) , l , w/neuron, edgecolor='k', facecolor='none')
                ax.add_patch(rect2)
                ax.add_patch(rect3)
                k = k + 1
                

    
            patch = patches.Rectangle((patchx+r*l/neuron, patchy+c*w/neuron),l/neuron , w/neuron, edgecolor='k', facecolor='k')
            ax.text(patchx+r*l/neuron+l/(neuron*12), patchy+c*w/neuron+w/neuron**2,str(int(r))+','+str(int(c)),color='white',fontsize='x-small')
            ax.add_patch(patch)
                    
            # Add the patch to the Axes
            ax.add_patch(rect)

            
            ax.set_xlim(properties.iloc[i,2:].index.min()*0.9,properties.iloc[i,2:].index.max()*1.1)
            ax.set_ylim(properties.iloc[:,2:].min().min()*0.9,properties.iloc[:,2:].max().max()*1.1)
                
            ax.set_xlabel('Raman shift (cm$^{-1}$)')
            ax.set_ylabel('Neuron Activation (a.u.)')

            wavenumber = pd.DataFrame(properties.iloc[i,2:].index).iloc[:,0]
            intensity = properties.iloc[i,2:].values
            
            intensity_b = (intensity-intensity.min())
  
            max_value.append(intensity_b.max())
            
            spectra.append(intensity_b/(intensity_b.max()-intensity_b.min()))
            
            smoothed_signal = smooth2(smooth,  intensity_b)
            smoothed_signal = (smoothed_signal-smoothed_signal.min())/(smoothed_signal.max()-smoothed_signal.min())
            
            local_max_index = find_local_max(smoothed_signal)
        
            stem = np.zeros(wavenumber.shape)
            stem[local_max_index] = 1  
            
            lorentz = []            

#            print(wavenumber[local_max_index])
#            plt.figure()
            
            for wav in wavenumber[local_max_index]:
        
                reg = fit_region(wavenumber,wav,10)
                
                try:
                    popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                                    reg, 
                                                                    smoothed_signal[reg.index],
                                                                    p0=[smoothed_signal[reg.index].max(), wav,  2/(np.pi*smoothed_signal[reg.index].max())])
                    
                    perr_1lorentz = np.sqrt(np.diag(pcov_1lorentz))
            
                    pars_1 = popt_1lorentz[0:3]
                    
                    if (abs(pars_1[-1])>100) | (pars_1[0]<0):
                        pass
                    else:

                        lorentz.append(_1Lorentzian(wavenumber,*popt_1lorentz))
                        peak_vector.append([label+'('+str(int(r))+' '+str(int(c))+')']+popt_1lorentz.tolist()+['('+str(int(r))+' '+str(int(c))+')'])
                        err.append(perr_1lorentz[0])

                except:
                    pass

        
    for peak, e in zip(peak_vector,err):
        peak.append(e)
    
    peak = pd.DataFrame(peak_vector,columns=['label','height','center','width','importance','err'] )
    
    spectra.append(wavenumber)  
    
    spectra = pd.DataFrame(spectra).T
#    print(spectra)
    spectra.columns = (peak['label']).unique().tolist()+['wavenumber']
    
    new_peak = []
    new_err = []
    
    for j in range(spectra.shape[1]-1):
        spectra.iloc[:,j] = spectra.iloc[:,j]*max_value[j]
        index = spectra.columns[j] == peak['label']
        new_peak.append(peak['height'].loc[index]*max_value[j])
        new_err.append(peak['err'].loc[index]*max_value[j])
        
    peak['Intensity (a.u.)'] = pd.DataFrame(np.concatenate([np.stack(a) for a in new_peak]))
    peak['err'] = pd.DataFrame(np.concatenate([np.stack(a) for a in new_err]))
        

    peak.to_csv('SOM_key'+label+'.csv',sep=';', index=False)
    
    if os.path.isfile('MASTERKEY.csv'):
        peak.to_csv('MASTERKEY.csv', sep=';', index=False,mode='a')
        clean_master('MASTERKEY.csv')
    else:
        peak.to_csv('MASTERKEY.csv', sep=';', index=False)
        


def ImportDataSOM(inputFilePath,windowmin,windowmax,zscore):
    wavenumber,spectra,label,spec_std = spec(inputFilePath)
    
    dataset = []
    labels = []

    for spe,lab in zip(spectra,label):   
        if zscore == 0:
            pass
        else:
            out = np.abs(stats.zscore(spe.T))
            spe = spe.T[(out < zscore).all(axis=1)].T       
            
        for i in range(spe.shape[1]):
            
            ynew = spe.iloc[:,i]
            base = als_baseline(ynew)
            ynew2 = pd.DataFrame(ynew - base)
            ynew2 = (ynew2-ynew2.min())/(ynew2.max()-ynew2.min())
            
#            ynew2.div(ynew2.sum(axis=1), axis=0)
            
            dataset.append(ynew2)
            labels.append(lab)
    
    
    intensity = pd.concat(dataset,axis=1).T
    intensity.columns = wavenumber
    intensity.index = labels
    
    if (windowmax !=0) & (windowmin != 0):
        region = (wavenumber > windowmin) & (wavenumber < windowmax)
        intensity = intensity.T.loc[wavenumber[region]].T
        
    intensity = pd.DataFrame(scale(intensity,axis=1),index=intensity.index,columns=intensity.columns)
    
    return intensity



def find_local_max(signal):
    dim = signal.shape[0]
    ans = np.zeros(dim)
    for i in range(dim):
        dxf = signal[i - 1] - signal[i]
        dxb = signal[i - 1] - signal[i - 2]
        ans[i] = 1.0 if dxf > 0 and dxb > 0 else 0.0
    return ans > 0.5



def impcolumn(data):
    peaks = []
    for lab in list(data.index.unique()):
        index = lab == data.index
        xx = data.columns
        yy = data[index].mean()
        
        yy = smooth2(7,yy)
        
        yy = (yy-yy.min())/(yy.max()-yy.min())
               
        local_max_index = find_local_max(yy)
           
        peaks.append(xx.values[local_max_index])
        
        finalp = sorted(set(list(itertools.chain.from_iterable(peaks))))

    
    return finalp
   

def LoadSOM(data,neuron):

    dataval = data.values
    
    som = MiniSom(neuron,neuron, data.shape[1], sigma=1.5, learning_rate=.5, 
                  neighborhood_function='gaussian', random_seed=0)
    
    som.pca_weights_init(dataval)
    som.train(dataval, neuron*3000, verbose=True)  # random training

    return som 



def PlotAvgSpec(data,windowmin,windowmax):
    classes = np.unique(data.index)
    
    NUM_COLORS = len(classes)

    cmaps = plt.get_cmap('gist_rainbow')

    colours = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    
    
    plt.figure()
    for lab,col in zip(classes,colours):
        index = lab == data.index
        xx = data.columns
        yy = data[index].mean()
        
        if (windowmin == 0) & (windowmax == 0):
            yy = (yy-yy.min())/(yy.max()-yy.min())
        
        plt.plot( xx , yy , label = lab,color=col)
    
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Relative Intensity (a.u.)')
    plt.legend(bbox_to_anchor=(0.1, 1.08), loc='upper left', fancybox=True, shadow=True,borderaxespad=0., ncol=3, fontsize=14)
    plt.show()


def rotate(point, angle):
    px, py = point

    qx =  np.cos(angle) *px - math.sin(angle) *py 
    qy =  np.sin(angle) *px + math.cos(angle) *py
    return qx, qy



def ReOrgScat(vec): 
    #max spherical number that can fit in an hexagon
#    maxr = 1.087
    rad = []
    cent = []
    particler = 0.056

    
    sort_vec = list(vec for vec,_ in itertools.groupby(sorted(vec)))
    
    sort_tup_vec = [tuple(l) for l in sort_vec]
    
    tup_vec = [tuple(l) for l in vec]
    
    d = dict()
    
    for k, v in groupby(tup_vec):
        d.setdefault(k, []).append(len(list(v)))


    for key,value in d.items():
        cent.append(key)
        rad.append(sum(value))


    cir_loc = []
    
    for p, r in zip(cent,rad):
        circles = circ.circlify([1]*r, show_enclosure=False,target_enclosure=circ.Circle(x=0, y=0, r=np.sqrt(r)*particler))
        for c in circles:
            x,y = rotate([c.x,c.y],np.pi/6)
            cir_loc.append([p,(x,y)])

    position_vector = []
    
    
    for i in range(len(tup_vec)):
        for j in range(len(cir_loc)):
            if tup_vec[i] == cir_loc[j][0]:
                position_vector.append((tup_vec[i][0]+cir_loc[j][1][0],tup_vec[i][1]+cir_loc[j][1][1]))
                del cir_loc[j]
                break

                        
    return position_vector    
    #the coordinate values can oscilate by .95 / np.sqrt(3)    
    
from collections import Counter


def ColorHexBins(plot_points,cla_col,wy,wx):
    final_c = []
    color_hex = []
    for pp,cc in zip(plot_points,cla_col):
        color_hex.append([pp[0],pp[1],cc[0],cc[1]])
    
    df_color_hex = pd.DataFrame(color_hex,columns=['pos x','pos y','label','color'])
    
    uni = (wx,wy)
    index1 =  df_color_hex['pos x']==uni[0]
    index2 =  df_color_hex['pos y']==uni[1]
    index = index1 & index2

    df_clean = df_color_hex[index].reset_index(drop=True) 
    
    counts = []
    
    for lab in df_clean['label'].unique():
        index3 = df_clean['label'] == lab
        counts.append([sum(index3),lab])

        
    counts = pd.DataFrame(counts,columns=['counts','label'])
    counts = counts.sort_values(by='counts')
    
    if counts.empty or counts['counts'].iloc[-1] == 0:
        final_c.append((0.5,0.5,0.5,1))
    else:
        index4 =  df_color_hex['pos x']==uni[0]
        index5 =  df_color_hex['pos y']==uni[1]
        index6 = df_color_hex['label']==counts['label'].iloc[-1]
        index7 = index4 & index5 & index6
        
        final_c.append(df_color_hex[index7]['color'].unique()[0])

    return final_c[0]
        



def HexagonaSOMPlot(som,data,neuron):
    
    target = list(data.index.values)
    le = preprocessing.LabelEncoder()
    le.fit(target)
    numtarget = le.transform(target)
    
    classes = np.unique(data.index)
    
#    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
    
#    colours = []
    
#    for i, x in enumerate(classes):
#        colours.append(cycle[i%len(cycle)])
    
    NUM_COLORS = len(classes)

    cmaps = plt.get_cmap('gist_rainbow')

    colours = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    
    
    dataval = data.values
    xx, yy = som.get_euclidean_coordinates()
    umatrix = som.distance_map()
    weights = som.get_weights()
    
    
    #code to align the hexagonals
    a = []
    
    for i in range(xx.shape[0]):
        b = []
        for j in range(xx.shape[1]):   
            if j % 2 == 0:
                b.append(0)
            else:
                b.append(0.5)
        a.append(b)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_aspect('equal')
    
   
        
    #enumerate gives you the count (cnt) and the interation value (x) represents each row
    cla_col = []
            
    plot_points = []
    for c , color in zip(classes,colours):
        idx_target = data.index.values == c

        
        for cnt, x in enumerate(dataval[idx_target]):
            # getting the winner
            w = som.winner(x)
            # place a marker on the winning position for the sample xx
            wx, wy = som.convert_map_to_euclidean(w) 
            
            if wy%2 == 0:
                wx = wx
            else:
                wx = wx+0.5
            
            wy = wy * np.sqrt(3) / 2
            
            plot_points.append([wx,wy])
            cla_col.append([c,color])
    

    
    vec = ReOrgScat(plot_points)
       
    
     # iteratively add hexagons
    
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            wy = yy[(i, j)] * np.sqrt(3) / 2
            wx = xx[(i, j)] + pd.DataFrame(a).values[(i,j)]
#            try:
                    
            hexa = RegularPolygon((wx, wy), 
                                 numVertices=6, 
                                 radius=.95 / np.sqrt(3),
                                 facecolor=cm.Blues(umatrix[i, j]), 
                                 alpha=.4, 
                                 linewidth=5,
                                 edgecolor= ColorHexBins(plot_points,cla_col,wy,wx))
            ax.add_patch(hexa)
#            except:
#                print(wx,wy)
    
    
    
    
    
    for (x0, y0) ,(c,color) in zip(vec,cla_col):
        plt.plot(x0 ,
             y0,
                'o',
                c=color,
                label = c )
            
    
    xrange = np.arange(weights.shape[0])
    yrange = np.arange(weights.shape[1])
    plt.xticks(xrange-.5, xrange)
    plt.yticks(yrange * np.sqrt(3) / 2, yrange)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
    cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues,  orientation='vertical', alpha=.4)
    cb1.ax.get_yaxis().labelpad = 16
    cb1.ax.set_ylabel('distance from neurons in the neighbourhood', rotation=270, fontsize=16)
    plt.gcf().add_axes(ax_cb)
    

    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.1, 1.08), loc='upper left', 
              fancybox=True, shadow=True,borderaxespad=0., ncol=3, fontsize=14)
    ax.axis('off')
    
    plt.show()




def NeuronActivationperWavePlot(som, data):   
    dataval = data.values
    columns = list(data.columns)
    win_map = som.win_map(dataval)
    size=som.distance_map().shape[0]
    
    #properties columns 0 and 1 are related to the positions, the rest are related to the weights
    #+2 to add space to insert the row and col place in the hexagonalbins
    
    properties = np.empty((size*size,2+dataval.shape[1]))
    properties[:]=np.NaN
    
    
    for row in range(0,size):
        for col in range(0,size):
            properties[size*row+col,0]=row
            properties[size*row+col,1]=col
    
    for position, values in win_map.items():
        properties[size*position[0]+position[1],0]=position[0]
        properties[size*position[0]+position[1],1]=position[1]
        properties[size*position[0]+position[1],2:] = np.mean(values, axis=0)
    
    
    B = ['row', 'col']
    B.extend(columns)
    properties = pd.DataFrame(data=properties, columns=B)
    
    row = math.ceil(math.sqrt(dataval.shape[1]))
    col = math.ceil(math.sqrt(dataval.shape[1]))
    
#    zmin=min(np.min(properties.iloc[:,2:]))
#    zmax=max(np.max(properties.iloc[:,2:]))
    
    #these plots show how the neurons reacted to the different Raman shifts.
    #for eg. how the 7x7 nuerons reacted to the 520cm-1 band?
    
    impeak = impcolumn(data)
    
    dim1 = int(np.sqrt(len(impeak)))+1
    
    if dim1*int(np.sqrt(len(impeak)))>=len(impeak):
        dim2 = int(np.sqrt(len(impeak)))
    else:
        dim2 = dim1
    

    plt.figure()
    #plt.imshow(z)
    
    gs = gridspec.GridSpec(dim1,dim2, wspace=0, hspace=0.5, top=0.95, bottom=0.05, left=0.17, right=0.845)
    
    all_heat = []
    
    #plot heatmap
    all_ax = []
    
    for i,col in enumerate(impeak):
        ax = plt.subplot(gs[i])
        z = properties.sort_values(by=['row', 'col'])[col].values.reshape(size,size) 
        all_heat.append(list(z.ravel())+[col])
#        z = ndimage.rotate(z, 90)
#        im = ax.imshow(z.T , cmap='cool' , interpolation = 'nearest' ,  vmin = np.nanmin(z) , vmax =np.nanmax(z))
        
        new_z = (z.T-np.nanmin(z.T))/(np.nanmax(z.T)-np.nanmin(z.T))
        im = ax.imshow(new_z , cmap='cool' , interpolation = 'nearest' ,  vmin = np.nanmin(new_z) , vmax =np.nanmax(new_z))
        
        
        
        
        ax.set_title(str(int(col)),fontsize=8.5,y=0.8)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.gca().invert_yaxis()
#        plt.colorbar()
#        plt.gca().invert_xaxis()
        plt.axis('off')
        all_ax.append(ax)
        
    plt.colorbar(im,ax=all_ax,cmap='cool')


    all_heat = pd.DataFrame(all_heat)

    res_heat = all_heat.iloc[:,:-1].max(axis=0).values.reshape(size,size)
    res_lab = []
    
    for i in range(all_heat.shape[1]-1):
        index = res_heat.ravel()[i] == all_heat.iloc[:,i]
        try:
            res_lab.append(all_heat.loc[index].iloc[:,-1].values[0])
        except:
            res_lab.append(np.nan)
            
    res_lab = np.array(res_lab).reshape(size,size)
            
    plt.figure()
    im = plt.pcolormesh(res_lab.T ,edgecolors='k', linewidth=2)

    plt.axis('off')
    
    values = res_heat.T.ravel()
    
    lab_cl = [x for x in list(set(res_lab.ravel())) if x == x]
    
    colors = [im.cmap(im.norm(value)) for value in lab_cl]
    # create a patch (proxy artist) for every color 
    patches1 = [ mpatches.Patch(color=colors[cnt], label=str(val)) for cnt,val in enumerate(lab_cl) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches1, bbox_to_anchor=(1/(size*2), 1.1), loc='upper left', fancybox=True, shadow=True,borderaxespad=0., ncol=size, fontsize=14)
    ax = plt.gca()
    ax.set_aspect('equal')
    



def Classification(som,data):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    X_train, X_test, y_train, y_test = train_test_split(data.values, data.index, stratify=data.index)    
    winmap = som.labels_map(X_train, y_train)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in X_test:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
            
    return classification_report(y_test, result)

#    
# file1 = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\saliva peaks from the literature.csv']
#lockgen = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key01st_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key02nd_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key03rd_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key04th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key05th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key06th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key04th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key05th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key06th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key07th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key08th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key09th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key10th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key11th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key13th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key14th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key15th_saliva.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keymilknm_.csv',
#          r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keysaliva_all..csv']
#
#keygen =[r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keymilknm_.csv',
#          r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keysaliva_all..csv']
#
#tol = 5

#dfs = [df1,df2]


def cleanfilepeak(dfs):
    new_dfs = []
    for df in dfs:
        
        if any(df.columns == 'height'):
            index1 = df['height']>0
        else:
            index1 =  [True]*df.shape[0]
            
        if (df.columns[-1] == 'Intensity (a.u.)'):
            index2 = df['importance']=='major'
            
        else:
            index2 = np.array([True]*df.shape[0])
            
        if any(df.columns == 'width'):
            index3 = df['width']>0
        else:
            index3 =  [True]*df.shape[0]
            
        index = index1 & index2 & index3

#decided not to group by on the chance that there are multiple values with the same wavenumber

        new_dfs.append(df[index])
    
    return new_dfs

#dfs = [newdf1,newdf2]
    
def comparefiles(dfs,tol):
    dataset = []
    #dfs[0]=df2 is related to the keygen which are the ones that we want to use as the denominator in the ration
    #dfs[1]=df1 are the lockgens which will be the numerator in the ration 
    df1 =  dfs[0]
    df2 =  dfs[1]
    

#    for k in df1['center'].unique(): 
    for k in df1['center']: 
        index = 0
        for l in df2['center']:
            if (l<k+tol) & (k-tol<l):
                
                if (df1.columns[-1] == 'Intensity (a.u.)') & (df2.columns[-1] == 'Intensity (a.u.)'):
                    dataset.append([np.mean([k,l]),
                    df1['label'][df1['center'] == k].values[0]+' & '+df2['label'][df2['center'] == l].values[0],
                    df2['Intensity (a.u.)'][df2['center']==l].values[0]/df1['Intensity (a.u.)'][df1['center']==k].values[0],
                    abs(df2['err'][df2['center']==l].values[0]*df1['Intensity (a.u.)'][df1['center']==k].values[0]-df2['Intensity (a.u.)'][df2['center']==l].values[0]*df1['err'][df1['center']==k].values[0])/(df1['Intensity (a.u.)'][df1['center']==k].values[0])**2])

                    
                elif list(str(df1.iloc[:,-1].values[0]))[0]=='Q':
                    dataset.append([np.mean([k,l]),
                                df1['label'][df1['center'] == k].values[0]+' & '+df2['label'][df2['center'] == l].values[0],
                                df1['importance'][df1['center']==k].values[0],
                                0])
                
                else:
                    dataset.append([np.mean([k,l]),
                                df1['label'][df1['center'] == k].values[0]+' & '+df2['label'][df2['center'] == l].values[0],
                                1,0])
                
                index = index + 1
        if index == 0:
            dataset.append([k,df1['label'][df1['center'] == k].values[0],0,0])
                
                
    return dataset
    

#        peak.to_csv('MASTERKEY.csv', sep=';', index=False,mode='a')
#        clean_master('MASTERKEY.csv')


#keygen = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keymilk_785nm_.csv']
#
#lockgen = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\key01st_saliva.csv',
#           r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keysaliva_785n.csv',
#           r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keymilk_785nm_.csv']
#
#tol = 5

def FilePeakMatching(keygen,lockgen,tol):
    
    all_f1 = []
    
    for file in keygen:
        df1 = pd.read_csv(file,sep=';')
        all_f1.append(df1)
        
    df1 = pd.concat([a for a in all_f1]).reset_index(drop=True)

    all_f2 = []
    
    for file in lockgen:
        df2 = pd.read_csv(file,sep=';')
        all_f2.append(df2)
    
    df2 = pd.concat([a for a in all_f2]).reset_index(drop=True)
    
    name1 = keygen[0].split('/')[-1]
    name2 = lockgen[0].split('/')[-1]
    
#    test
#    name1 = keygen[0].split('\\')[-1]
#    name2 = lockgen[0].split('\\')[-1]
    
    newdf1,newdf2 = cleanfilepeak([df1,df2])
    data = comparefiles([newdf1,newdf2],tol)
    dataset = pd.DataFrame(data,columns=['center','label','ratio','err'])
    dataset.to_csv(name1+'_'+name2+'.csv',index=False)
    
    dataset2,code = orderfiles(dataset)
    
    shape = len(df1['label'].unique())
    
    plotpoints(dataset2,shape)
    centerpointplots(dataset2,shape,code)


    return 1

   
def plotpoints(dataset,shape):
    
    name0 = dataset['label'].unique()[0].split(' ')[0]
    
    NUM_COLORS = len(dataset['label'].unique())//shape
    
    cmaps = plt.get_cmap('gist_rainbow')
    
    colours = [cmaps(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]*shape
    
    
    plt.figure()
    for name,colo in zip(dataset['label'].unique(),colours):
        if name.split(' ')[0] != name0:
            plt.figure()
            index = name == dataset['label']
            plt.plot(dataset['center'][index],dataset['ratio'][index],'-o',label=name,c=colo)
            name0 = name.split(' ')[0]
        else:
            index = name == dataset['label']
            plt.plot(dataset['center'][index],dataset['ratio'][index],'-o',label=name,c=colo)
        
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel('I/I0')
        plt.legend(loc='best', prop={'size':12},frameon=False)
        plt.show()
    


def collapsebull(lis,a):
    flist = []
    lis = pd.DataFrame(lis)
    for i in range(lis.shape[1]):
        if a == 'any':
            flist.append(any(lis.iloc[:,i]))
        elif a == 'all':
            flist.append(all(lis.iloc[:,i]))
        
    return flist
 
# shape = len(df1['label'].unique())

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def checklabelloc(data,c):
    label = []   
    for cs in c:
        index = data['center']==cs
        label.append(data['label'][index].values[0])
    
    return len(label) == len(set(label))
    

from sklearn import preprocessing




def orderfiles(dataset):
    
    order = []
    for cnt, lab in enumerate(dataset['label'].unique()):
        split = lab.split(' ')
        if split[0] == split[-1]:
#            a = cnt
            order.append(0)
        else:
            order.append(cnt+1)
#            if cnt == 0:
#                order.append(np.nan)
#            else:
#                order.append(cnt)
            #the +1 is to make sure that the 0 position is allocated for the same label case
               
#    order = [a if math.isnan(x) else x for x in order]

    
    dataset['new label'] = np.nan
    
    for lab, o in zip(dataset['label'].unique(),order):
        index = lab == dataset['label']
        dataset['new label'][index] = int(o) 
    
    plotlab = pd.DataFrame([dataset['label'].unique(),pd.Series(order).unique()]).T
    plotlab.columns = ['label','order']
    plotlab = plotlab.sort_values(by='order').reset_index(drop=True)
    
    code = [plotlab['label'].values,plotlab['order'].values]
    
    return dataset,code
    


def centerpointplots(dataset,shape,code):
    
    # clusters = cluster(sorted(dataset['center'].values),5)
    iterate = int(len(dataset['label'].unique())/shape) 

    
    for i in range(shape):
        indexes = []
        for lab in dataset['label'].unique()[iterate*i:iterate*(i+1)]: 
            indexes.append(lab == dataset['label'])
        
        cindex = collapsebull(indexes,'any')

        
        data = dataset[cindex].reset_index(drop=True)
        limminx = sorted(data['new label'].unique())[0]
        limmaxx = sorted(data['new label'].unique())[-1] 
        
        clusters = cluster(sorted(data['center'].values),2)
        
        for c in clusters:

#            if len(c)>np.sqrt(iterate):
#            if len(c)==iterate and checklabelloc(data,c):
#                print(c)

            x = []
            y = []
            z = []
            for cs in c:
                index = data['center']==cs
                x.append(data['new label'][index].values[0])
                y.append(data['ratio'][index].values[0])
                z.append(data['err'][index].values[0])
            
            if limminx in x and limmaxx in x:
                plt.figure()
            
                new_coord =[]
                
                x = pd.Series(x)
                y = pd.Series(y)
                z = pd.Series(z)
                
                for xs in x.unique():
                    index = x == xs
                    new_coord.append([xs,y[index].mean(),z[index].min()])
                    
                df = pd.DataFrame(new_coord)
                df.columns = ['label','ratio','err']
                df = df.loc[pd.to_numeric(df['label'], errors='coerce').sort_values().index]
                
                
                
                fit = linear_fit
                x = df['label']
                xx = np.linspace(0,len(x)-1,len(x))
                y = df['ratio']
                yerr = df['err']
                
    #                print(df)
                
                 
                try:   
                    pars, cov = curve_fit(f=fit, xdata=xx, ydata=y, p0=[1, 1])
                    stdevs = np.sqrt(np.diag(cov))
                    res = y - fit(xx, *pars)
                    ss_res = np.sum(res**2)
                    ss_tot = np.sum((y-np.mean(y))**2)
                    r_squared = 1 - (ss_res / ss_tot)
    
                    plt.errorbar(x,y,yerr=yerr,fmt='s',capsize=5,label = str(round(cs,1)))
                    plt.plot(xx, fit(xx, *pars), color='r',linestyle='--', linewidth=2)
                    plt.text(xx.max()*0.8, y.max()*0.925,'$R^{2}=$'+str(round(r_squared,2)),fontsize='xx-small')
                    plt.text(xx.max()*0.6, y.max()*0.9,'ratio = '+str(round(pars[0],2))+'$x + $'+str(round(pars[1],2)),fontsize='xx-small')
                    
                except:
                    plt.errorbar(x,y,yerr=df['err'],fmt='s',capsize=5,label = str(round(cs,1)))
    
    
                plt.xlabel('Sample')
                plt.ylabel('Ratio')
                plt.xticks(range(iterate))
                plt.text(xx.min(),y.min(),''.join([str(a)+' : '+str(b)+'\n' for a,b in zip(code[0],code[1])]),fontsize='xx-small')
                plt.legend(loc='upper right', prop={'size':12},frameon=False)
                plt.show()















#SOM adapted from
#@misc{vettigliminisom,
#  title={MiniSom: minimalistic and NumPy-based implementation of the Self Organizing Map},
#  author={Giuseppe Vettigli},
#  year={2018},
#  url={https://github.com/JustGlowing/minisom/},
#}    
    
    
    
#############################################################################################################################################
#############################################################################################################################################   
#############################################################################################################################################
    #TESTING CENTER###############
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

#windowmin=700
#windowmax=1700
#keygen = inputFilePath
#smooth=7
#asymmetry_param = 0.05
#smoothness_param = 1000000
#max_iters = 10
#conv_thresh =0.00001
#zscore=3
#inv=0  
#neuron = 3
#
#data = ImportDataSOM(keygen,windowmin,windowmax,zscore)
#som = LoadSOM(data,neuron)
#HexagonaSOMPlot(som,data)
#NeuronActivationperWavePlot(som, data)
#text = Classification(som,data)


#zscore = 0
#asymmetry_param = 0.05
#smoothness_param = 1000000
#max_iters = 10
#conv_thresh = 0.00001
#inv = 0 
#wavenumber,z,label,sstd = spec(inputFilePath)
    
    
    

#fig = plt.figure(figsize=(9,9/1.618))
##x = wavenumber
##y = pd.DataFrame(new_spec).T 
##y2 = dc
#
#wav,z,lab,zstd = spec(inputFilePath)
#z = separate_stack(z)
#z,zstd = STDS(z)
#spectra = diff(z,zstd)
#StackPlot(wav,spectra,zstd,lab,fig)

    
 #x = wavenumber
#y = intensity   
#y2 = pd.DataFrame()        
#x = wavenumber
#y = intensity 
#x = wavenumber
#y1 = intensity 
#y2 = pd.DataFrame()
#zscore = 3
#inv = 1

#inputFilePath = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 3\data\Al_All_static_325-1490_1s_10%_1acc_785nm_01.txt',
#                 r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 3\data\Al_Kaf_static_325-1490_1s_10%_1acc_785nm_01.txt']

#spectra = pd.concat(spec,axis=1).T
#spectra_std = pd.DataFrame(spec_std[0]).T

#spectra = pd.concat(spec,axis=1).T
#spectra_std = pd.concat(spec_std,axis=1).T

#spectra = z
#spectra_std = zstd
        
#newspectra=pd.DataFrame(new_spec).T    
#y = pd.DataFrame(spec).T
#for all the files together: stacked and 3d
#y = z
#
#asymmetry_param = 0.05
#smoothness_param = 1000000
#max_iters = 10
#conv_thresh =0.00001
#zscore=0
#files=inputFilePath
#inv=0
#norm = f_dataset
#labels = f_target
#windowmin = 0
#windowmax = 0
#
#finalDf = df
#targets = target
#stylestring = 'binary'
#dim = 2

#targets = target
#finalDf = df
#dim = 2
#
#files = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 7\MIP Melezitose\80nm comparison\7)80nm_st95.txt'
#    ,
#         r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 7\MIP Melezitose\80nm comparison\8)80nm_mel_st95.txt']
#
#files = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 3\data\Sta_0.001mM_1100_spec_3s_10%_6acc_785nm_01.txt',
#        r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 3\data\Sta_0.01mM_1100_spec_3s_10%_6acc_785nm_01.txt',
#        r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 3\data\Sta_0.1mM_1100_spec_3s_10%_6acc_785nm_01.txt',
#        r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 3\data\Sta_1mM_1100_spec_3s_10%_6acc_785nm_01.txt',
#        r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 3\data\Sta_10mM_1100_spec_3s_10%_6acc_785nm_01.txt']
##
#smooth=7
#asymmetry_param = 0.05
#smoothness_param = 1000000
#max_iters = 10
#conv_thresh =0.00001
#zscore=0
#inv=0   
#    
##    
#files = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\mikes data\BCG_Copy.txt',
#         r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\mikes data\C.glutimacum_785nm_06.02.2020.txt',
#         r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\mikes data\R Erythropolis 21.11.2019.txt']
#inputFilePath = files
#wavenumber,z,label,sstd = spec(inputFilePath)
#empty = pd.DataFrame()
#z = separate_stack(z)
#z,sstd = STDS(z)
#spectra = diff(z,empty)
#fig = plt.figure(figsize=(9,9/1.618))
#StackPlot(wavenumber, spectra, empty,label, fig)
#    
    
#files = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\zscore stu\bad\BCG_Copy.txt',
#         r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\zscore stu\bad\C.glutimacum_785nm_06.02.2020.txt',
#         r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\zscore stu\bad\R Erythropolis 21.11.2019.txt']

 
 
 
 
 
 
 
 
