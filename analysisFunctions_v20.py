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

bin_colours = ['r','orange',
              'g','lime',
              'b','cyan',
              'gold','yellow',
              'indigo','violet',
              'pink','magenta',
              'maroon','chocolate',
              'grey','darkkhaki']
 
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

#def smooth2(t, signal):
#    dim = signal.shape[0]
#    U, s = get_line_laplacian_eigen(dim)
#    return matrix_exp_eigen(U, -s, t, signal)


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


#inputFilePath = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 7\MIP Melezitose\ST95\80nm_st95_785nm_10%_3s_3acc_1000cm_100map.txt'
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
        
        #works for testing
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


#fig = plt.figure(figsize=(9,9/1.618))
#x = wavenumber
#y = intensity   
#y2 = pd.DataFrame()

def StackPlot(x,y,y2,label,fig): 
#    print(label)
    if y2.empty:
        for i in range(y.shape[1]): 
            plt.plot(x, y.iloc[:,i] ,label=label[i],lw=2)    
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel(' Intensity (a.u.)')
        plt.legend(loc='best', prop={'size':12},frameon=False)
        fig.show()
    else:
        for n in range(y.shape[1]):
            plt.plot(x, y.iloc[:,n] ,label=label[n],lw=2)
            plt.fill_between(x, y.iloc[:,n] - y2.iloc[:,n] ,y.iloc[:,n] + y2.iloc[:,n] , color='orange', alpha=0.5)
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel(' Intensity (a.u.)')
        plt.legend(loc='best', prop={'size':12},frameon=False)
        fig.show()

#still to fix!
      
#ax = plt.axes(projection = '3d')      
#x = wavenumber
#y = intensity   
#y2 = pd.DataFrame()
        
def ThreeDPlot(x,y,y2,label,ax):
    if y2.empty:
        x = x.values
        spec = y
        n_max = y.shape[1]
        for n in range(n_max):
            z = spec.iloc[:,n]
            y = np.array([n]*z.shape[0])
            
            ax.plot(x,  [n]* z.shape[0],  z, c='k', lw=4,zorder=abs(n-n_max))
            ax.plot(x,  [n]* z.shape[0],  z, lw=1,zorder=abs(n-n_max))
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
            ax.plot(x,  [n]* z.shape[0],  z, lw=1,zorder=abs(n-n_max))
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
        
#x = wavenumber
#y = intensity 

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


#x = wavenumber
#y1 = intensity 
#y2 = pd.DataFrame()
    
    
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


#zscore = 3
#inv = 1


def spec(inputFilePath):
    
    spec = []
    spec_std = []
    label = []
    for file in inputFilePath:
        spec.append(ImportData(file)[1])
#        if ImportData(file,zscore,inv)[1].std(axis=1).isnull().all():
#            spec_std = pd.DataFrame()
#        else:           
        spec_std.append(ImportData(file)[1])
        
        #works for testing
#        label.append(file.split('\\')[-1][:11])
        
        label.append(file.split('/')[-1][:11])
    wavenumber = ImportData(file)[0]
    
    return (wavenumber,pd.DataFrame(spec).T,label,pd.DataFrame(spec_std).T)

#spectra = pd.DataFrame(spec).T
#spectra_std = pd.DataFrame(spec_std).T

def diff(spectra,spectra_std):
    if spectra_std.empty:     
        new_spec = []
        for i in range(spectra.shape[1]):
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
        return pd.DataFrame(new_spec).T
    else:
        
        miss_elem = list(set([i for i in spectra.columns]) - set([i for i in spectra_std.columns]))
        inter = pd.DataFrame(0, index=np.arange(spectra_std.shape[0]),columns = miss_elem)
        spectra_std = pd.concat([spectra_std,inter],axis=1)
        spectra_std.reindex(sorted(spectra_std.columns), axis=1)

        
        new_spec = []
        for i in range(spectra.shape[1]):
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
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.title(label)
    
    else:
        plt.figure(figsize=(9,9/1.618))
        ax = sns.heatmap(ynew.T,xticklabels=xticklabels,vmin=0,cbar=True)
        plt.ylim(-1,y.shape[1]+1)
        plt.xlim(-1,y.shape[0]+1)
        ax.set_xticks(xticks)
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

#y = pd.DataFrame(spec).T

def BaselineNAVG(y,asymmetry_param, smoothness_param, max_iters, conv_thresh):
#    y = pd.DataFrame(y)
    y = y.dropna(axis='columns')
    a = []
    for i in range(y.shape[1]):
        a.append(y.iloc[:,i].values-als_baseline(y.iloc[:,i],asymmetry_param, smoothness_param, max_iters, conv_thresh))    
    z = pd.DataFrame(np.transpose(a))
    z.columns = y.columns
    
    return z    

#for all the files together: stacked and 3d


def separate_stack(y):
    a = []
    b = []
    
    for j in range(y.shape[1]):
        z = y[j][0]
        z = z.dropna(axis='columns')
        for i in range(z.shape[1]):
            a.append(z.iloc[:,i])    
            b.append(i)
    sp = pd.DataFrame(np.transpose(a),columns=np.transpose(b))
    
    return sp

def BaselineNAVGS(y,asymmetry_param, smoothness_param, max_iters, conv_thresh):
    a = []
    b = []
    
    for j in range(y.shape[1]):
        z = y[j][0]
        z = z.dropna(axis='columns')
        for i in range(z.shape[1]):
            a.append(z.iloc[:,i].values-als_baseline(z.iloc[:,i],asymmetry_param, smoothness_param, max_iters, conv_thresh))    
            b.append(i)
    sp = pd.DataFrame(np.transpose(a),columns=np.transpose(b))
    
    return sp
    
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

#asymmetry_param = 0.05
#smoothness_param = 1000000
#max_iters = 10
#conv_thresh =0.00001
#z=0
#files=inputFilePath
#inv=0

   
def sortData(files,asymmetry_param ,smoothness_param ,max_iters ,conv_thresh,zscore,inv):
    f_dataset = []
    f_target = []
    for file in files:
        target = []
        dataset = []
            
        wavenumber, intensity, n_points, n_raman,label,flag = ImportData(file)       
        if intensity.shape[1] == 0:
            flag = 0
        else:
            
            baseline = BaselineNAVG(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)
            norm = NormalizeNAVG(baseline,zscore,inv)
            
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

#norm = f_dataset
#labels = f_target
#windowmin = 0
#windowmax = 0

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

    if flag == 0:
        
        position = []
    
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
        
            
        fig = plt.figure(figsize=(9,9/1.618))
        ax = fig.add_subplot(1,1,1)
        plt.xlabel(str(info[0]))
        plt.ylabel(str(info[1]))
        
        colours = []
        
        for i in range(len(list(np.unique(target)))):
            colours.append(cycle[i%len(cycle)])
            
#        colours = bin_colours
            
        for label,color in zip(list(np.unique(target)),colours):
            indicesToKeep = df['sample'] == label
    
            new_x = df.loc[indicesToKeep, 'principal component 1']
            new_y = df.loc[indicesToKeep, 'principal component 2']
    
            ax.scatter(new_x, new_y, s =100,alpha=1,label=label,marker='x')
    
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(loc='best',frameon=False)
            
            position.append([[(new_x.max()-new_x.min())/2,(new_y.max()-new_y.min())/2],
                       (new_x.max()-new_x.min()),
                       (new_y.max()-new_y.min()),label])
            
            plt.show()
        
        
    else:
        
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
        
        position = []
            
        fig = plt.figure(figsize=(9,9/1.618))
        ax = fig.add_subplot(1,1,1)
        plt.xlabel(str(info[0]))
        plt.ylabel(str(info[1]))
        
        colours = []
        
        for i in range(len(list(np.unique(target)))):
            colours.append(cycle[i%len(cycle)])
            
#        colours = bin_colours
            
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


#finalDf = df
#targets = target
#stylestring = 'binary'
#dim = 2


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


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score

def dist_k_calc(finalDf,stylestring,targets,position,colours,dim):
    
#    colours = bin_colours
    
    if stylestring == 'binary':
        
        param = 2
        fig, axs = plt.subplots(len(list(np.unique(targets)))//param//3 if len(list(np.unique(targets)))//param>=3 else 1, 3 ,figsize=(5*len(list(np.unique(targets)))//param,5))
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
                

                axs[n].contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
                axs[n].plot(PC1,PC2,'o',label = list(np.unique(targets))[n*param],color = colours[n*param])
                axs[n].plot(PC10,PC20,'o',label = list(np.unique(targets))[n*param+1],color = colours[n*param+1])
                axs[n].legend(loc='best',frameon=False,fontsize='x-small')
                axs[n].text(xx.min()*0.9, yy.min()*0.9, 'Sep : %0.1f %% +/- %0.1f %%' % (scores.mean()*100, scores.std()*100), size=15)


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
                        z1 = dist_sug.iloc[i,2]
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
    
    colours = []
    for n in range(len(names)):
        colours.append(cycle[n%len(cycle)])
        
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

    if flag == 0:
    
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
        

        ax.view_init(30,-45)
        ax.set_xlabel(str(info[0]),linespacing=5)
        ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))           
        ax.set_ylabel(str(info[1]),linespacing=5)
        ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))            
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(str(info[2]),rotation=90,linespacing=5)
        ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
        
        colours = []
        
        for i in range(len(list(np.unique(target)))):
            colours.append(cycle[i%len(cycle)])
            
#        colours = bin_colours
            
        for label,color in zip(list(np.unique(target)),colours):
            indicesToKeep = df['sample'] == label
    
            new_x = df.loc[indicesToKeep, 'principal component 1']
            new_y = df.loc[indicesToKeep, 'principal component 2']
            new_z = df.loc[indicesToKeep, 'principal component 3']
    
            ax.scatter(new_x, new_y,new_z, s=100, alpha=0.6, label=label,marker='o')
    
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(loc='best',frameon=False)
            
            plt.show()
        
        
    else:
        
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
            
        ax.view_init(30,-45)
        ax.set_xlabel(str(info[0]),linespacing=5)
        ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))           
        ax.set_ylabel(str(info[1]),linespacing=5)
        ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))            
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(str(info[2]),rotation=90,linespacing=5)
        ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
        
        positionz = []
        colours = []
        
        for i in range(len(list(np.unique(target)))):
            colours.append(cycle[i%len(cycle)])
            
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

def smooth2(t, signal):
    dim = signal.shape[0]
    U, s = get_line_laplacian_eigen(dim)
    return matrix_exp_eigen(U, -s, t, signal)

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
    
    df.to_csv('MASTER.csv', sep=';', index=False)
    
    return df


#peak
def clean_master_df(peak):
    df = peak.drop_duplicates()
    index = df.columns[0] != df.iloc[:,0]
    df = df[index]
    
    return df

def peakfinder(files,smooth=7,asymmetry_param = 0.05,smoothness_param = 1000000,max_iters = 10,conv_thresh =0.00001,zscore=0,inv=0):

    peak_vector = []
    max_value = []
    spectra = []
    
    if len([files])==1:
        files = [files]
        
    for file in files:

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
                lorentz_peak_1 = _1Lorentzian(reg, *pars_1)
                
                ax4.plot(wavenumber,_1Lorentzian(wavenumber,*popt_1lorentz))
                ax4.fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),
                                     _1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
                
                
    
                lorentz.append(_1Lorentzian(wavenumber,*popt_1lorentz))
                peak_vector.append([label]+popt_1lorentz.tolist()+['major'])
            
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
        

        ax4.plot(wavenumber, smoothed_signal ,'o',color='g')
        ax4.plot(wavenumber[local_max_index],smoothed_signal[local_max_index],'x',color='r')
        
            
        ax1.plot(wavenumber,residual_1lorentz,'o',color='b')
        ax1.plot(wavenumber[local_max_index2],residual_1lorentz[local_max_index2],'x',color='orange')
        
#        ax4.plot(wavenumber,-residual_1lorentz,'o')
        ax1.plot(wavenumber[local_max_index3],residual_1lorentz[local_max_index3],'x',color='orange')
        
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
                    lorentz_peak_1 = _1Lorentzian(reg, *pars_1) 
                    
                    
                    ax1.plot(wavenumber,_1Lorentzian(wavenumber,*popt_1lorentz))
                    ax1.fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),
                                     _1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
                    
                    lorentz2.append(_1Lorentzian(wavenumber,*popt_1lorentz))
                    peak_vector.append([label]+popt_1lorentz.tolist()+['minor positive'])
                
                except:
                    pass
                
                        
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
                    lorentz_peak_1 = _1Lorentzian(reg, *pars_1) 
    
                    
                    ax1.plot(wavenumber,-_1Lorentzian(wavenumber,*popt_1lorentz))
                    ax1.fill_between(wavenumber, _1Lorentzian(wavenumber,*popt_1lorentz).min(),
                                     -_1Lorentzian(wavenumber,*popt_1lorentz), alpha=0.5)
                    
                    lorentz_neg.append(_1Lorentzian(wavenumber,*popt_1lorentz))
    
                    peak_vector.append([label]+[a*b for a,b in zip(popt_1lorentz.tolist(),[-1,1,1])]+['minor negative'])
                
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
    
    peak = pd.DataFrame(peak_vector,columns=['label','height','center','width','importance'] )
    
    spectra.append(wavenumber)  
    
    spectra = pd.DataFrame(spectra).T
    spectra.columns = peak['label'].unique().tolist()+['wavenumber']
    
    new_peak = []
    
    for i in range(spectra.shape[1]-1):
        spectra.iloc[:,i] = spectra.iloc[:,i]*max_value[i]
        index = spectra.columns[i] == peak['label']
        new_peak.append(peak['height'].loc[index]*max_value[i])
        
    peak['Intensity (a.u.)'] = pd.DataFrame(np.concatenate([np.stack(a) for a in new_peak]))
        

    peak.to_csv('key'+label+'.csv',sep=';', index=False)
    
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
    
    z = np.abs(stats.zscore(df['center']))
    df = df[(z < 1)]
    
#    z2 = np.abs(stats.zscore(df['Intensity (a.u.)']))
#    df = df[(z2<1)]

    string = ['pM','nM','uM','mM','M']
    mult = [10**-12,10**-9,10**-6,10**-3,1] 
    
    conc_s = []
    dil = df['label'].values
    

    for d in dil:
        for s in string:
#            match = re.search("[-+]?\d*\.\d+|\d+"+s, d)
            match = re.findall(r"[-+]?\d*\.\d+|\d+"+s,d)
#            print(s,match)
            if match:
                conc_s.append(match[0])

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
    



def plot_dilution(peak,spectra,number):
    if number ==0:
            
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
            col = label == spectra.iloc[:,:-1].columns
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
            col = label == spectra.iloc[:,:-1].columns
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

        for target in df['sample'].unique().tolist():
            index = df['sample'] == target
            axbig3.plot(df['principal component 1'][index],df['principal component 2'][index],'o',label=target)
        
        axbig3.fill_between(np.linspace(0, df['principal component 1'].max()*1.1,10),0,(df['principal component 2'].max()*1.1),alpha=0.2)
        axbig3.fill_between(np.linspace(-df['principal component 1'].max()*1.1,0,10),0,(df['principal component 2'].max()*1.1),alpha=0.2)
        axbig3.fill_between(np.linspace(-df['principal component 1'].max()*1.1,0,10),0,(-df['principal component 2'].max()*1.1),alpha=0.2)
        axbig3.fill_between(np.linspace(0, df['principal component 1'].max()*1.1, 10),0,(-df['principal component 2'].max()*1.1),alpha=0.2)
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
            
        for ax in list(chain(*axs[2:,2:])):
            ax.remove()
            
        for j in range(0,2):
            for ax in axs[j,2:]:
                ax.remove()
            for ax in axs[2:,j]:
                ax.remove()
    
            
        axbig = fig.add_subplot(gs[:2,:2])
        axbig2 = fig.add_subplot(gs[2:,2:])
        
        axsma = fig.add_subplot(gs[0,2:])
        axsma2 = fig.add_subplot(gs[1,2:])
        
        axsma3 = fig.add_subplot(gs[2,:2])
        axsma4 = fig.add_subplot(gs[3,:2])
        
        combax = [axsma,axsma2,axsma3,axsma4]
 
        axbig2.plot(wavenumber,loadings['LPC1'],label = 'LPC1')
        axbig2.plot(wavenumber,loadings['LPC2'],label = 'LPC2')
        axbig2.legend(loc='best', frameon=False,prop={'size': 10})  
        
        axbig.grid(True)
        axbig2.grid(True)
        

        for target in df['sample'].unique().tolist():
            index = df['sample'] == target
            axbig.plot(df['principal component 1'][index],df['principal component 2'][index],'o',label=target)
        
        axbig.fill_between(np.linspace(0, df['principal component 1'].max()*1.1, 10),0,(df['principal component 2'].max()*1.1),alpha=0.2)
        axbig.fill_between(np.linspace(-df['principal component 1'].max()*1.1,0, 10),0,(df['principal component 2'].max()*1.1),alpha=0.2)
        axbig.fill_between(np.linspace(-df['principal component 1'].max()*1.1,0, 10),0,(-df['principal component 2'].max()*1.1),alpha=0.2)
        axbig.fill_between(np.linspace(0, df['principal component 1'].max()*1.1, 10),0,(-df['principal component 2'].max()*1.1),alpha=0.2)
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
        

def loadingpeak(maps,smooth,
                asymmetry_param, 
                smoothness_param, 
                max_iters, conv_thresh,
                zscore,inv,  Q, windowmin,
                windowmax, dim=2,flag=1):
    
    wavenumber,norm,labels,flag =  mapload(maps,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv,flag=1)
    pca,wavenumber,df = pcafit(norm, wavenumber,labels, windowmin,  windowmax)
    loadings = plotload(pca,wavenumber,dim)
    plotprinc(df,loadings,wavenumber,pca,Q,smooth)
    



#lockgens = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 7\MIP Melezitose\80nm comparison\7)80nm_st95.txt',
#    r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 7\MIP Melezitose\80nm comparison\8)80nm_mel_st95.txt'
#    ]
#keygen = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\5) 10mM_Mel.csv']
#
#    
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

#lock is a map
#key is a generated csv
    
def peakmatching(keygen,lockgens,tol = 5):
    key = pd.read_csv(keygen[0],sep=';')
    
    fig, ax = plt.subplots(len(lockgens), 1, sharex='all', sharey='all',figsize=(9,len(lockgens)*9/1.618))
    i=0
        
    
    for lockgen in lockgens:
                   
        peak, spectra = peakfinder(lockgen,smooth=7,asymmetry_param = 0.05,smoothness_param = 1000000,max_iters = 10,conv_thresh =0.00001,zscore=0,inv=0)
        peak = peak[peak['Intensity (a.u.)']>0]
        lock,conc_s = clean_up_rep(peak)
         
         
        key = key[key['Intensity (a.u.)']>0]
#        key = key[key['label']==key['label'].iloc[-1]]
    
         
        match = []
         
        for k in key['center'].unique():
            for l in lock['center'].unique():
                if (l<k+tol) & (k-tol<l):
                    match.append([k,key['label'][key['center'] == k].values[0]])
            
    
        val = pd.DataFrame(match).iloc[:,0]
        nam = pd.DataFrame(match).iloc[:,1]
           
        lab = []
        
        m = list(set(val.tolist()))
        
        for ms in m:
            lab.append(nam[val==ms].values[0])
            
        cluster_m = cluster(m,tol)
        
        dataset = []
         
        for c,nam in zip(cluster_m,lab):
            dataset.append([np.mean(c),nam])
            
        wavenumber = spectra.iloc[:,-1]
        
        
        xnew = np.linspace(key['center'].min(),key['center'].max(),1000)
        
        sim_plot = []
        for cent,inten,wid in zip(key['center'].tolist(),key['Intensity (a.u.)'].tolist(),key['width'].tolist()):
            sim_plot.append(_1Lorentzian(xnew,inten,cent,wid))
        
        sim_plot = pd.DataFrame(sim_plot).sum()
    
        if len(lockgens) == 1:
            ax.plot(wavenumber,spectra.iloc[:,:-1], label = spectra.columns[0])
            for data in dataset:
                ax.fill_between(np.linspace(data[0]-tol,data[0]+tol,10),spectra.iloc[:,:-1].max()[0],color='yellow')
                ax.text(data[0]-tol,spectra.iloc[:,:-1].max()[0]*0.8,data[1],rotation=90,fontsize='xx-small')
            
            ax.legend(loc='best',frameon=False)
            ax.set_xlabel('Raman shift $(cm^{-1})$')
            ax.set_ylabel('Intensity (a. u.)')
    #        plt.ylim(0,3000)
            plt.show()

        else:
        
            ax[i].plot(wavenumber,spectra.iloc[:,:-1], label = spectra.columns[0])
            for data in dataset:
                ax[i].fill_between(np.linspace(data[0]-tol,data[0]+tol,10),spectra.iloc[:,:-1].max()[0],color='yellow')
                ax[i].text(data[0]-tol,spectra.iloc[:,:-1].max()[0]*0.8,data[1],rotation=90,fontsize='xx-small')
                
            ax[i].legend(loc='best',frameon=False)
            ax[i].set_xlabel('Raman shift $(cm^{-1})$')
            ax[i].set_ylabel('Intensity (a. u.)')
            plt.show()
            i = i+1
  
#keygen = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\5) 10mM_Mel.csv']
#lockgen = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\loadingsall1)80nm_au.t.csv']
    

     

def peakloadmatching(keygen,lockgen,tol=5):
    key = pd.read_csv(keygen[0],sep=';')
    lock = pd.read_csv(lockgen[0], sep=';')
    
    key = key[key['Intensity (a.u.)']>0]
#    key = key[key['label']==key['label'].iloc[-1]]
    
    lock = lock[lock['height']>0]
    
    match = []
     
    for k in key['center'].unique():
        for l in lock['center'].unique():
            if (l<k+tol) & (k-tol<l):
                match.append([k,key['label'][key['center'] == k].values[0]])
        

    val = pd.DataFrame(match).iloc[:,0]
    nam = pd.DataFrame(match).iloc[:,1]
       
    lab = []
    
    m = list(set(val.tolist()))
    
    for ms in m:
        lab.append(nam[val==ms].values[0])
        
    cluster_m = cluster(m,tol)
    
    dataset = []
     
    for c,nam in zip(cluster_m,lab):
        dataset.append([np.mean(c),nam])

    xnew = np.linspace(key['center'].min(),key['center'].max(),1000) 
    
    sim_plot = []
    for cent,inten,wid in zip(key['center'].tolist(),key['Intensity (a.u.)'].tolist(),key['width'].tolist()):
        sim_plot.append(_1Lorentzian(xnew,inten,cent,wid))
    
    sim_plot = pd.DataFrame(sim_plot).sum()

    xnew2 = np.linspace(lock['center'].min(),lock['center'].max(),1000)
    
    sim_plot2 = []
    sim_plot_sep = []

    for imp in lock['importance'].unique().tolist():
        aux = []
        index = lock['importance'] == imp
        for cent,inten,wid in zip(lock['center'][index].tolist(),lock['height'][index].tolist(),lock['width'][index].tolist()):
            sim_plot2.append(_1Lorentzian(xnew2,inten,cent,wid))
            aux.append(_1Lorentzian(xnew2,inten,cent,wid))
        sim_plot_sep.append(pd.DataFrame(aux).sum().values)
    
    sim_plot2 = pd.DataFrame(sim_plot2).sum()
    sim_plot_sep = pd.DataFrame(sim_plot_sep)
    sim_plot_sep['importance'] = pd.DataFrame(lock['importance'].unique().tolist())
    
    

    fig, ax2 = plt.subplots(figsize=(9,9/1.618)) 
    ax2.plot(xnew2,sim_plot2,label='Q sum')
    soma = sim_plot_sep.iloc[:,:-1].max(axis=1).sum() 
    a = 0
    Qs = sim_plot_sep['importance']
    
    for imp in Qs.tolist():
        ax2.fill_between(xnew2,sim_plot_sep.iloc[:,:-1][Qs==imp].values[0]+a,a,label=imp)
        a = a + sim_plot_sep.iloc[:,:-1][Qs==imp].values[0]
    for data in dataset:
        ax2.fill_between(np.linspace(data[0]-tol,data[0]+tol,10),soma,color='yellow',alpha=1/4)
        ax2.text(data[0]-tol,soma*0.8,data[1],rotation=90,fontsize='xx-small')
    ax2.set_xlabel('Raman shift $(cm^{-1})$')
    ax2.set_ylabel('Loadings (a.u.)') 
    plt.xticks(np.arange(xnew2.min()-1,xnew2.max()+1,50),rotation=90)
    ax2.legend(loc='best',frameon=False,ncol=5,fontsize=12)
    plt.show()
    
    
    
#    
#    fig, (ax1,ax2) = plt.subplots(2,1,sharex='all', sharey='all', figsize=(9,9/1.618)) 
#    
#    ax1.plot(xnew,sim_plot,label='Lorentz GenKey')
#    ax2.plot(xnew2,sim_plot2*sim_plot.max(),label='Q sum')
#    for data in dataset:
#        ax2.fill_between(np.linspace(data[0]-tol,data[0]+tol,10),sim_plot.max(),color='yellow',alpha=0.4)
#        ax2.text(data[0]-tol,sim_plot.max()*0.8,data[1],rotation=90,fontsize='xx-small')
#    a = 0
#    for imp in sim_plot_sep['importance'].tolist():
#        ax2.fill_between(xnew2,sim_plot_sep.iloc[:,:-1][sim_plot_sep['importance']==imp].values[0]*sim_plot.max()+a,a,alpha=0.5,label=imp)
#        a = a + sim_plot_sep.iloc[:,:-1][sim_plot_sep['importance']==imp].values[0]*sim_plot.max()
#    
#
#   
#
#    ax2.set_xlabel('Raman shift $(cm^{-1})$')
#    ax2.set_ylabel('Loadings (a.u.)') 
#    ax1.set_ylabel('Intensity (a.u.)') 
#    ax1.legend(loc='best',frameon=False,fontsize=12)
#    ax2.legend(loc='best',frameon=False,fontsize=12)
#    plt.show()
#    
























