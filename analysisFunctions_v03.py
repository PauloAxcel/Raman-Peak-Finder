import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Gets rid of some warnings in output text
import warnings
warnings.filterwarnings("ignore")

from scipy.linalg import solveh_banded
from scipy import signal
from sklearn.decomposition import PCA
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from scipy import stats
import nestle

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


#for file in inputFilePath:
#    label = []
#    
#    label.append(file.split('\\')[-1][:11])
#
#    print(file.split('\\')[-1][:11])
    
#inputFilePath = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\1st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt'
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
        flag = 1
    return (wavenumber, intensity, n_points, n_raman, label,flag)


def StackPlot(x,y,y2,label,fig): 
#    print(label)
    if y2.empty:
        for i in range(len(label)): 
            plt.plot(x, y.iloc[:,i] ,label=label[i],lw=2)    
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel(' Intensity (a.u.)')
        plt.legend(loc='best', prop={'size':12},frameon=False)
        fig.show()
    else:
        for i in range(len(label)): 
            plt.plot(x, y.iloc[:,i] ,label=label[i],lw=2)  
        for n in range(y.shape[1]):
            plt.fill_between(x, y.iloc[:,n] - y2.iloc[:,n] ,y.iloc[:,n] + y2.iloc[:,n] , color='orange', alpha=0.5)
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel(' Intensity (a.u.)')
        plt.legend(loc='best', prop={'size':12},frameon=False)
        fig.show()

#still to fix!
      
#x = a[0]
#y = b
#y2 = a[3]
#label = a[2]  
#ax = plt.axes(projection = '3d')      
        
        
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
    plt.plot(x, y1+y1.min()*n ,label=label,lw=2)    
    plt.fill_between(x, y1- y2 +y1.min()*n, y1 + y2 +y1.min()*n, color='orange', alpha=0.5)
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel(' Intensity (a.u.)')
    plt.legend(loc='best', prop={'size':12},frameon=False)
    fig.show()


def spec(inputFilePath):
    spec = []
    spec_std = []
    label = []
    for file in inputFilePath:
        spec.append(ImportData(file)[1].mean(axis=1))
        spec_std.append(ImportData(file)[1].std(axis=1))
        label.append(file.split('/')[-1][:11])
    wavenumber = ImportData(file)[0]
#    label = ImportData(file)[4]
    
    return (wavenumber,pd.DataFrame(spec).T,label,pd.DataFrame(spec_std).T)

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
    y_avg = y.mean(axis=1)
    y_std = y.std(axis=1)    
    return y_avg, y_std
                 
def Baseline(y,asymmetry_param, smoothness_param, max_iters, conv_thresh):
    if pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    return y_avg - als_baseline(y_avg,asymmetry_param, smoothness_param, max_iters, conv_thresh)


def BaselineNAVG(y,asymmetry_param, smoothness_param, max_iters, conv_thresh):
#    y = pd.DataFrame(y)
    a = []
    for i in range(y.shape[1]):
        a.append(y.iloc[:,i].values-als_baseline(y.iloc[:,i],asymmetry_param, smoothness_param, max_iters, conv_thresh))    
    z = pd.DataFrame(np.transpose(a))
    z.columns = y.columns
    
    return z    
    
def Normalize(y):
    if pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    return ((y_avg-y_avg.min())/(y_avg.max()-y_avg.min()))


def NormalizeNAVG(y):
    a = []
    for i in range(y.shape[1]):
        a.append((y.iloc[:,i].values-y.iloc[:,i].min())/(y.iloc[:,i].max()-y.iloc[:,i].min()))
    z = pd.DataFrame(np.transpose(a))
    z.columns = y.columns
    
    return z

def NormalizeNSTD(y,ystd):
    a = []
    for i in range(y.shape[1]):
        a.append((ystd.iloc[:,i].values/y.iloc[:,i].max()))
    z = pd.DataFrame(np.transpose(a))
    z.columns = y.columns
    
    return z
#
#asymmetry_param = 0.05
#smoothness_param = 1000000
#max_iters = 10
#conv_thresh =0.00001
#z=0
#files=inputFilePath

   
def sortData(files,asymmetry_param ,smoothness_param ,max_iters ,conv_thresh,z):
    f_dataset = []
    f_target = []
    for file in files:
        target = []
        dataset = []
            
        wavenumber, intensity, n_points, n_raman,label,flag = ImportData(file)         
        baseline = BaselineNAVG(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)
        norm = NormalizeNAVG(baseline)
        
        where_are_NaNs = np.isnan(norm)
        if sum(where_are_NaNs.values[0])>0:
            flag = 0
            norm[where_are_NaNs] = 0
        
        for i in range(norm.shape[1]):
            target.append(label)
            dataset.append(norm.iloc[:,i])
        if z == 0:
            for data,text in zip(dataset,target):
                f_dataset.append(data)
                f_target.append(text)
                flag = 1
        else:
            dataset2 = pd.DataFrame(dataset)
            target2 = pd.DataFrame(target)
            out = np.abs(stats.zscore(dataset2))
            
            for value,text in zip(dataset2[(out<z).all(axis=1)].values,target2[(out<z).all(axis=1)].values):
                for data, string in zip(value,text):                    
                    f_dataset.append(value)
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
    
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
        
        position = []
            
        fig = plt.figure(figsize=(9,9/1.618))
        ax = fig.add_subplot(1,1,1)
        plt.xlabel(str(info[0]))
        plt.ylabel(str(info[1]))
        
        colours = []
        
        for i in range(len(list(np.unique(target)))):
            colours.append(cycle[i%len(cycle)])
            
        for label,color in zip(list(np.unique(target)),colours):
            indicesToKeep = df['sample'] == label
    
            new_x = df.loc[indicesToKeep, 'principal component 1']
            new_y = df.loc[indicesToKeep, 'principal component 2']
    
            ax.scatter(new_x, new_y, s =100,alpha=1,label=label,marker='x')
    
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(loc='best',frameon=False)
            
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
            
        for label,color in zip(list(np.unique(target)),colours):
            indicesToKeep = df['sample'] == label
            
            gmm = GaussianMixture(n_components=1).fit(pd.concat([df.loc[indicesToKeep, 'principal component 1']
                       , df.loc[indicesToKeep, 'principal component 2']],axis=1).values)
        
        
           
            for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
                
                
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
                position.append([pos,np.sqrt(width/2),np.sqrt(height/2),label])
                
                range1s = (((new_x < pos[0]+np.sqrt(width/2)) & (new_x > pos[0]-np.sqrt(width/2))) | 
                        ((new_y < pos[1]+np.sqrt(height/2)) & (new_y > pos[1]-np.sqrt(height/2))))
                ax.scatter(new_x[range1s], new_y[range1s], s =100,alpha=1,label=label,marker='x')
            
            #    plot_gmm(gmm)
            
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(loc='best',frameon=False)
            
            plt.show()

    
def LoadingsPlot2D(wavenumber,pca):
    plt.figure(figsize=(9,9/1.618))
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_))
    loadings = loadings.iloc[:,:2]
    loadings.columns = ['LPC1','LPC2']
    plt.plot(wavenumber,savgol_filter(loadings['LPC1'],11,3),label='LPC1')
    plt.plot(wavenumber,savgol_filter(loadings['LPC2'],11,3),ls='--',label='LPC2')
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
            x[i,j], y[i,j], z[i,j] = ell.ctr + np.dot(ell.axes,
                                                      [x[i,j],y[i,j],z[i,j]])

    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=color, alpha=0.2)



def PCAPlot3D(df,target,info,flag,ax):    

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
        
        colours = []
        
        for i in range(len(list(np.unique(target)))):
            colours.append(cycle[i%len(cycle)])
            
        for label,color in zip(list(np.unique(target)),colours):
            indicesToKeep = df['sample'] == label
            new_x = df.loc[indicesToKeep, 'principal component 1']
            new_y = df.loc[indicesToKeep, 'principal component 2']
            new_z = df.loc[indicesToKeep, 'principal component 3']
            
            gmmf = GaussianMixture(n_components=1).fit(pd.concat([df.loc[indicesToKeep, 'principal component 1'],
                                  df.loc[indicesToKeep, 'principal component 2']
                               , df.loc[indicesToKeep, 'principal component 3']],axis=1).values)  
            
            centers = gmmf.means_
            U, s, Vt = np.linalg.svd(gmmf.covariances_)
            widths, heights,zeds = 2 * np.sqrt(s[0])
            

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
            ells = nestle.bounding_ellipsoid(concat, pointvol)
                   
            ax.scatter(new_x[range1 & range2 & range3], new_y[range1 & range2 & range3],new_z[range1 & range2 & range3], s=100, alpha=0.6, color=color ,label=label,marker='o')
#            for ell in ells:
            plot_ellipsoid_3d(ells, ax, color)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(loc='best',frameon=False)
        
        plt.show()



def LoadingsPlot3D(wavenumber,pca):
    plt.figure(figsize=(9,9/1.618))
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_))
    loadings = loadings.iloc[:,:3]
    loadings.columns = ['LPC1','LPC2','LPC3']
    plt.plot(wavenumber,savgol_filter(loadings['LPC1'],11,3),label='LPC1')
    plt.plot(wavenumber,savgol_filter(loadings['LPC2'],11,3),ls='--',label='LPC2')
    plt.plot(wavenumber,savgol_filter(loadings['LPC3'],11,3),ls=':',label='LPC3')
    plt.legend(loc='best',frameon=False)
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Loadings')
    plt.show()





















