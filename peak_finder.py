# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:12:26 2018

@author: user
"""
import pylab
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import solveh_banded
from scipy.signal import savgol_filter
#import scipy.fftpack
#from lmfit.models import LorentzianModel
import pickle # for loading pickled test data
import matplotlib
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks_cwt
from scipy import signal
from pandas.plotting import scatter_matrix
from pandas.plotting import andrews_curves
from pandas.plotting import autocorrelation_plot
from pandas.plotting import radviz
import seaborn as sns
from sklearn import mixture
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

from matplotlib.patches import Ellipse

font = {'family' : 'Arial',
        'size'   : 22}

plt.rc('font', **font)


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



# From Signal1DUtils
def find_local_max(signal):
    dim = signal.shape[0]
    ans = np.zeros(dim)
    for i in range(dim):
        dxf = signal[i - 1] - signal[i]
        dxb = signal[i - 1] - signal[i - 2]
        ans[i] = 1.0 if dxf > 0 and dxb > 0 else 0.0
    return ans > 0.5

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


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
#    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


 


files = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 7\MIP Melezitose\Melezitose\coffeering\785nm\1) 1uM_Mel_al_785nm_10%_3s_3acc_1000cm_100map_coffeering1.txt',
r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 7\MIP Melezitose\Melezitose\coffeering\785nm\2) 10uM_Mel_al_785nm_10%_3s_3acc_1000cm_100map_coffeering.txt',
r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 7\MIP Melezitose\Melezitose\coffeering\785nm\3) 100uM_Mel_al_785nm_10%_3s_3acc_1000cm_100map_coffeering.txt',
r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 7\MIP Melezitose\Melezitose\coffeering\785nm\4) 1mM_Mel_al_785nm_10%_3s_3acc_1000cm_100map_coffeering.txt',
r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 7\MIP Melezitose\Melezitose\coffeering\785nm\5) 10mM_Mel_al_785nm_10%_3s_3acc_1000cm_100map_coffeering.txt'
]



def ImportData(inputFilePath,zscore,inv):
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
        
        #works for testing
#        label = inputFilePath.split('\\')[-1][:11]

            
        for i in range(df.shape[0]//n_points):
            ind_df = df['Intensity'][i*n_points:n_points*(i+1)].reset_index(drop=True)
            ind_df = ind_df.rename('spectrum '+str(i+1))
            dataset.append(ind_df)
    
        intensity = pd.concat(dataset,axis=1)
#        print(intensity.shape)
        
        if zscore == 0:
            pass
        else:
            out = np.abs(stats.zscore(intensity.T))
            
            if inv == 0:
                intensity = intensity.T[(out < zscore).all(axis=1)].T
#                print(intensity.shape)
            else:
                intensity = intensity.T[(out > zscore).all(axis=1)].T       
#                print(intensity.shape)
        
        if intensity.shape[1] == 1:
            flag = 0
        else:
            flag = 1
        
    return (wavenumber, intensity, n_points, n_raman, label,flag)


#
#asymmetry_param = 0.05
#smoothness_param = 1000000
#max_iters = 10
#conv_thresh =0.00001
#z=0
#inv=0
    

import scipy as scipy

def Normalize(y):
    if pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    return ((y_avg-y_avg.min())/(y_avg.max()-y_avg.min()))


def Baseline(y,asymmetry_param, smoothness_param, max_iters, conv_thresh):
    if pd.DataFrame(y).shape[1]==1 and isinstance(y, pd.DataFrame):
        y_avg = y.iloc[:,0]
    elif pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    return y_avg - als_baseline(y_avg ,asymmetry_param, smoothness_param, max_iters, conv_thresh)

def _1Lorentzian(x, amp, cen, wid):
    return amp*wid**2/((x-cen)**2+wid**2)
  
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

from matplotlib import gridspec               

for file in files:
    wavenumber, intensity, n_points, n_raman, label,flag = ImportData(file,0,0)
    
#    norm = Normalize(intensity)


    intensity_b = Baseline(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)
    
    smoothed_signal = smooth2(7,  intensity_b)
    smoothed_signal = Normalize(smoothed_signal)
#    smoothed_signal=(smoothed_signal-smoothed_signal.min())/(smoothed_signal.max()-smoothed_signal.min())
    
#    plt.plot(wavenumber, smoothed_signal)

    # find local maximum minimum indexes
    
    local_max_index = find_local_max(smoothed_signal)
    

    # draw local maximum
    
    stem = np.zeros(wavenumber.shape)
    stem[local_max_index] = 1  
    
    fig = plt.figure(figsize=(9,9/1.618))
    gs = gridspec.GridSpec(3,1, height_ratios=[1,0.25,1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
        
    ax1.plot(wavenumber,smoothed_signal,'o')
    ax1.plot(wavenumber[local_max_index],smoothed_signal[local_max_index],'x')
    
    lorentz = []
    
    for wav in wavenumber[local_max_index]:

        reg = fit_region(wavenumber,wav,10)
        
        try:
            popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                            reg, 
                                                            smoothed_signal[reg.index],
                                                            p0=[1, wav, 5])
        except:
            pass
        
        perr_1lorentz = np.sqrt(np.diag(pcov_1lorentz))

        pars_1 = popt_1lorentz[0:3]
        lorentz_peak_1 = _1Lorentzian(reg, *pars_1)
        
#        residual_3lorentz = smoothed_signal - _1Lorentzian(wavenumber,*popt_1lorentz)
        
        ax1.plot(wavenumber,_1Lorentzian(wavenumber,*popt_1lorentz))
        ax1.fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),
                         _1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
        
        lorentz.append(_1Lorentzian(wavenumber,*popt_1lorentz))
        
    lorentz_all = Normalize(Baseline(pd.DataFrame(lorentz).sum()
                           ,asymmetry_param, smoothness_param, max_iters, conv_thresh))
    residual_1lorentz = smoothed_signal - lorentz_all 
    ax2.plot(wavenumber, residual_1lorentz,'o',color='b')
    
    ax3.plot(wavenumber,smoothed_signal,'o')
    ax3.fill_between(wavenumber, lorentz_all,0, alpha=0.5)
    
    
    local_max_index2 = find_local_max(residual_1lorentz.values)
    

    # draw local maximum
    
    stem2 = np.zeros(wavenumber.shape)
    stem2[local_max_index2] = 1 
    
    
    fig = plt.figure(figsize=(9,9/1.618))
    gs = gridspec.GridSpec(3,1, height_ratios=[1,0.25,1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
        
    ax1.plot(wavenumber,residual_1lorentz,'o')
    ax1.plot(wavenumber[local_max_index2],residual_1lorentz[local_max_index2],'x')
    
    lorentz2 = []
    
    for wav in wavenumber[local_max_index2]:

        reg = fit_region(wavenumber,wav,10)
        
        try:
            popt_1lorentz, pcov_1lorentz = scipy.optimize.curve_fit(_1Lorentzian, 
                                                            reg, 
                                                            residual_1lorentz[reg.index],
                                                            p0=[residual_1lorentz.max(), wav, 5])
        except:
            pass
        
        perr_1lorentz = np.sqrt(np.diag(pcov_1lorentz))

        pars_1 = popt_1lorentz[0:3]
        lorentz_peak_1 = _1Lorentzian(reg, *pars_1)
        
#        residual_3lorentz = smoothed_signal - _1Lorentzian(wavenumber,*popt_1lorentz)
        
        ax1.plot(wavenumber,_1Lorentzian(wavenumber,*popt_1lorentz))
        ax1.fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),
                         _1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
        
        lorentz2.append(_1Lorentzian(wavenumber,*popt_1lorentz))
    
    lorentz_all2 = Baseline(pd.DataFrame(lorentz2).sum()
                       ,asymmetry_param, smoothness_param, max_iters, conv_thresh)
    
    residual_1lorentz2 = residual_1lorentz - lorentz_all2
    
    ax2.plot(wavenumber, residual_1lorentz2,'o',color='b')
#    
#    ax3.plot(wavenumber,residual_1lorentz2,'o')
#    ax3.fill_between(wavenumber, lorentz_all2,0, alpha=0.5)
    ax3.plot(wavenumber, Normalize(lorentz_all+lorentz_all2))
    ax3.plot(wavenumber,Normalize(intensity_b),'o')
    
    
#    ax3.plot(wavenumber,lorentz_all)
    
    
    
    
#    for wav in wavenumber[local_max_index]:
#        index = wavenumber == wav
#        plt.figure()
#        plt.plot(wavenumber[wavenumber==wav-25:wavenumber==wav+25])
#        

    
    a.append([ wavenumber[local_max_index] ,smoothed_signal[local_max_index]])
    
    
    df_1_2 = df_1_2.append({'center': xData[local_max_index] , 'height' : smoothed_signal[local_max_index]} , ignore_index=True)

#####REMOVE STRING ROW + PREPARE DATA#####
    
center1 = np.hstack(df_1_2.iloc[:,0])
height1 = np.hstack(df_1_2.iloc[:,1])

a=[]

for i in range(len(height1)):
    a.append([center1[i] , height1[i]])

df1_2 = pd.DataFrame(a)

X = df1_2.values


#################################
#################################
##BEST CLUSTERING FOR PEAK DATA##
#################################
#################################

#peak = 25

#
#n_components = np.arange(100, 600, 50)
#models = [GMM(n, covariance_type='full', random_state=0)
#          for n in n_components]
#aics = [model.fit(X).aic(X) for model in models]
#plt.plot(n_components, aics)
#
#min_y = min(aics)  # Find the maximum y value
#min_x = n_components[aics.index(min_y)] 
#print(min_x)

#use min(aics) for the best value of peaks??
#number of peaks should be the maximum peaks obtained in the peak finding section after baseline

a = []

for i in range(n_parts):
    a.append(df_1_2.iloc[i,:][0].size)
    
peaks = int(np.round(np.max(a)))

gmm = GMM(n_components=peaks, covariance_type='full', random_state=0)
plot_gmm(gmm, X)


center = gmm.means_[:,0]
height = gmm.means_[:,1]
standard_dev = gmm.covariances_

dftest = pd.read_csv(r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\MT_Measurement techniques\MI\experimental work\RAMAN\MI completed\data anaylisis with python\raman peaks.csv',sep=';',header=None)
dftest = dftest.dropna(axis='columns')
dftest.columns = ['bond','range-','range+','average','intensity']


#####################################################
####REMOVE ELEMENTS THAT ARE NOT GIVEN###############
#####################################################

#string = 'CHSSiAuBNO'

import re

string = re.findall('[A-Z][^A-Z]*',string)

dftest_aux = pd.DataFrame()

all_bonds = dftest['bond'].astype(str)


for i in range(len(all_bonds)):
    all_bonds [i] = re.findall('[A-Z][^A-Z]*',all_bonds[i])


       
for i in range(dftest.shape[0]):
    if any(set(list(string)) & set(all_bonds[i])):
       dftest_aux = dftest_aux.append({ 'bond' : dftest.iloc[i,0] , 'range-' : dftest.iloc[i,1] , 'range+' : dftest.iloc[i,2], 'average' : dftest.iloc[i,3] ,'intensity' : dftest.iloc[i,4] } , ignore_index=True)
    if not string:
       dftest_aux = dftest

      
#################################################################       
#######################DFTEST_AUX COLUMNS CHANGE ORDER...........
#################################################################
       
a = []

#tol = 50

for i in range(dftest_aux.shape[0]):
    for j in range(len(center)):
        if dftest_aux.iloc[i,3] - tol <= center[j] <= dftest_aux.iloc[i,3] + tol:
            a.append([dftest_aux.iloc[i,0],center[j],height[j],dftest_aux.iloc[i,4]])
            
            
df_peaks = pd.DataFrame(a)
df_peaks.drop_duplicates(keep='first',inplace=True)

###############################################
####ARE THE ELEMENTS PRESENT IN THE SAMPLE?####
###############################################


       
df_1.columns = ['center','intensity']

a=[]
a = np.array_split(df_1, df_1[df_1.iloc[:,0] == df_1.iloc[0,0]].shape[0])
a =  np.hstack(a)

dfB = pd.DataFrame(a)
dfB = dfB.T
dfB = dfB.drop_duplicates(keep=False)
dfB.reset_index()
dfB_mean =  dfB.mean(axis=0)
dfB_std = dfB.std(axis=0)

dfB_all = pd.concat([df_1.iloc[:dfB.iloc[0].size,0] , dfB_mean],axis=1)
x = dfB_all.iloc[:,0]
y = dfB_all.iloc[:,1]


    ######PLOT DATA-BASELINE########
    
base = als_baseline(yData)
diff = abs(base-yData)
    


    # Signal smoothing 
smoothed_signal = smooth2(7,  diff)
smoothed_signal=smoothed_signal-min(smoothed_signal)


##############################################################################
#########WRITE LABELS ON THE GRAPH WITH RESPECTIVE BONDS######################
##############################################################################


fig, ax = plt.subplots()
#
plt.plot(dfB_all.iloc[:,0],dfB_all.iloc[:,1],label = 'raw data')

plt.fill_between(dfB_all.iloc[:,0], dfB_all.iloc[:,1] - dfB_std, dfB_all.iloc[:,1] + dfB_std ,facecolor='lightgrey', alpha=0.5)

#plt.plot(dfB_all.iloc[:,0],smoothed_signal,label='baseline+smoothing' )
plt.legend(loc='best')

from adjustText import adjust_text


texts = []
texts_aux = []

###########################################################
#MAKE RELATION BETWEEN THE PEAKS EXTRACTED FROM THE GAUSSIAN WITH THE REAL AVERAGE SPECTRA#####
###############################################################################################

for i in range(df_peaks.shape[0]):
    for j in range(dfB_all.shape[0]):
        if abs(dfB_all.iloc[j,0]-df_peaks.iloc[i,1])<2:
            texts.append([dfB_all.iloc[j,0] , dfB_all.iloc[j,1] , df_peaks.iloc[i,0]])
            break

df_text = pd.DataFrame(texts)

center_f = np.ndarray.tolist(df_text.iloc[:,0].values)
height_f = np.ndarray.tolist(df_text.iloc[:,1].values)
text_f = np.ndarray.tolist(df_text.iloc[:,2].values)

for x, y, s in zip(center_f, height_f, text_f):
    texts_aux.append(plt.text(x, y, s))

############ALWAYS REMOVE X=X AND Y=Y###########
    
adjust_text(texts_aux, autoalign='y',
            only_move={'points':'y', 'text':'yx'}, force_points=0.15,
            arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.show()

##############################################################################
#########MAKE TABLE RELATING THE INTENSITY WITH THE LABEL STRENGTH############
##############################################################################

df_peaks = df_peaks.reset_index(drop=True)
df_peaks.columns = ['bond','center','intensity','strenght']

intensity = df_peaks['intensity'].values

a=[]

dist = intensity.max()-intensity.min()


for i in range(df_peaks.shape[0]):
    if df_peaks['strenght'][i] == 'vs' or df_peaks['strenght'][i] == 's':
        a.append([df_peaks['bond'][i] , round(100*abs(abs(intensity[i]-intensity.max())-dist)/dist,1)])
    if df_peaks['strenght'][i] == 'm' or df_peaks['strenght'][i] == 'sm' or df_peaks['strenght'][i] == 'mw':
        a.append([df_peaks['bond'][i] , round(100*abs(abs(intensity[i]-intensity.mean())-dist)/dist,1)])
    if df_peaks['strenght'][i] == 'w' or df_peaks['strenght'][i] == 'vw':
        a.append([df_peaks['bond'][i] , round(100*abs(abs(intensity[i]-intensity.min())-dist)/dist,1)])
 
df_bond = pd.DataFrame(a)
df_bond.columns = ['Bonds','Accuracy']
#df_bond['Bonds'] = '"' + df_bond['Bonds'] + '"'
#df_bond.to_csv('bonds.csv' ,sep=';',mode='a',index=False)


#from pandas.plotting import table
#
#fig2, ax2 = plt.subplots()    
#
#table(ax2, np.round(df_bond, 2),loc='best')
#
#fig2.tight_layout()
#        
#plt.show()

########################################################
###########COLOR AND SCROLLING TABLE####################
#########################################################
#
##import plotly.plotly as py
#import plotly as py
#from plotly.offline import init_notebook_mode
#import plotly.graph_objs as go
#
#py.offline.init_notebook_mode(connected=True)
#
#
#table_trace=dict(type = 'table',
#                 columnwidth= [3]+[3],
#                 columnorder=[0, 1],
#                 header = dict(height = 50,
#                               values = [['<b>Bonds</b>'], ['<b>Accuracy</b>']],
#                               line = dict(color='rgb(50,50,50)'),
#                               align = ['center']*2,
#                               font = dict(color=['rgb(45,45,45)']*2, size=14),
#                               fill = dict( color = 'rgb(235,235,235)' )
#                              ),
#                 cells = dict(values = [df_bond['Bonds'], df_bond['Accuracy']],
#                              line = dict(color='#506784'),
#                              align = ['center']*5,
#                              font = dict(color=['rgb(40,40,40)']*2, size=12),
#                              format = [None, ",.1f"],
#                              #prefix = [None],
#                              #suffix=[None],
#                              height = 30,
#                              fill = dict(color=['rgb(245,245,245)',#unique color for the first column
#                                                ['rgba(0,250,0, 0.8)' if val>=50 else 'rgba(250,0,0, 0.8)' for val in df_bond['Accuracy']] ]
#                               #the cells in the second column colored with green or red, according to their values
#                                         )
#                             )
#                  )
#                              
#                              
#layout = dict(width=600, height=1000, autosize=True, 
#              title='Table with cells colored<br>according to their values', showlegend=False)
#fig= go.Figure(data=[table_trace], layout=layout)
#
##py.sign_in('PauloAxcel', 'RvNoVXvmjKow70WKmDEK')
#py.offline.plot(fig, filename = 'table')
#













