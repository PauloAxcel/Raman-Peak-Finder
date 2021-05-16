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
import statsmodels.stats.api as sms     # t-test
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

    
    dataset = []
    labels = []
    for file in inputFilePath:
        

        df = pd.read_csv(file, sep = '\s+',  header = None,  skiprows = 1, names = ['X','Y','Wavenumber', 'Intensity'])
        n_points = sum((df.iloc[0,1]==df.iloc[:,1][df.iloc[0,0]==df.iloc[:,0]]))
        n_raman = df.shape[0]//n_points
        wavenumber = df['Wavenumber'][0:n_points]
        
        
        xnew = np.linspace(int(wavenumber.min()), int(wavenumber.max()),  int(round(wavenumber.max()-wavenumber.min())+1))  
#        xnew = np.array(range(int(wavenumber.min()+1),int(wavenumber.max())-1))
        label = file.split('\\')[-1][:15]

            
        for i in range(df.shape[0]//n_points):
            ind_df = df['Intensity'][i*n_points:n_points*(i+1)].reset_index(drop=True)
            
            spl = make_interp_spline(sorted(wavenumber),ind_df[::-1],k=3)
            
            ynew = pd.DataFrame(spl(xnew))
            
            
            
            base = als_baseline(ynew.iloc[:,0])
            
            ynew2 = pd.DataFrame(ynew.iloc[:,0] - base)
            
            ynew2 = (ynew2-ynew2.min())/(ynew2.max()-ynew2.min())
            
#            ynew2.div(ynew2.sum(axis=1), axis=0)
            
            dataset.append(ynew2)
            labels.append(label)
    
    
    intensity = pd.concat(dataset,axis=1).T
    intensity.columns = xnew
    intensity.index = labels
        
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
    som.train(dataval, 10000, verbose=True)  # random training

    return som

from itertools import chain

def PlotAvgSpec(data):
    classes = np.unique(data.index)
    plt.figure()
    for lab in classes:
        index = lab == data.index
        xx = data.columns
        yy = data[index].mean()
      
        yy = (yy-yy.min())/(yy.max()-yy.min())
        
        plt.plot( xx , yy , label = lab)
    
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
    










def HexagonaSOMPlot(som,data):
    
    target = list(data.index.values)
    le = preprocessing.LabelEncoder()
    le.fit(target)
    numtarget = le.transform(target)
    
    classes = np.unique(data.index)
    
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
    
    colours = []
    
    for i, x in enumerate(classes):
        colours.append(cycle[i%len(cycle)])
    
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
    
    # iteratively add hexagons
    
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            wy = yy[(i, j)] * np.sqrt(3) / 2
            wx = xx[(i, j)] + pd.DataFrame(a).values[(i,j)]
            
            hexa = RegularPolygon((wx, wy), 
                                 numVertices=6, 
                                 radius=.95 / np.sqrt(3),
                                 facecolor=cm.Blues(umatrix[i, j]), 
                                 alpha=.4, 
                                 edgecolor='gray')
            ax.add_patch(hexa)
        
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


import math
from scipy import ndimage
    
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
    
    zmin=min(np.min(properties.iloc[:,2:]))
    zmax=max(np.max(properties.iloc[:,2:]))
    
    #these plots show how the neurons reacted to the different Raman shifts.
    #for eg. how the 7x7 nuerons reacted to the 520cm-1 band?
    
    impeak = impcolumn(data)
    
    dim1 = int(np.sqrt(len(impeak)))+1
    
    if dim1*int(np.sqrt(len(impeak)))>=len(impeak):
        dim2 = int(np.sqrt(len(impeak)))
    else:
        dim2 = dim1
    
    import matplotlib.gridspec as gridspec
    
    plt.figure()
    #plt.imshow(z)
    
    gs = gridspec.GridSpec(dim1,dim2, wspace=0, hspace=0.5, top=0.95, bottom=0.05, left=0.17, right=0.845)
    
    
    for i,col in enumerate(impeak):
        ax = plt.subplot(gs[i])
        z = properties.sort_values(by=['row', 'col'])[col].values.reshape(size,size) 
#        z = ndimage.rotate(z, 90)
        ax.imshow(z.T , cmap='cool' , interpolation = 'nearest' , vmin = zmin , vmax = zmax)
        
        ax.set_title(str(int(col)),fontsize=8.5,y=0.8)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.gca().invert_yaxis()
#        plt.gca().invert_xaxis()
        plt.axis('off')




def classify(som, data):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = som.labels_map(X_train, y_train)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


from matplotlib.patches import Patch
import matplotlib as mpl

def most_relevant_feature(som):
    size=som.distance_map().shape[0]
    W = som.get_weights()
    impeak = impcolumn(data)
    Z = np.zeros((size, size))
    plt.figure(figsize=(8, 8))
    position = []
    for i in np.arange(som._weights.shape[0]):
        for j in np.arange(som._weights.shape[1]):
            feature = np.argmax(W[i, j , :])
            peak = int(data.iloc[:,feature].name)
            position.append([j, i,peak])
            
    df = pd.DataFrame(position,columns=['x','y','peak'])
    pivot = pd.pivot_table(df,values='peak',index=['x'],columns=['y'])
    
    
    peak_elements = df['peak'].drop_duplicates().values
    
    cmap = plt.cm.cool  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    # Manually specify colorbar labelling after it's been generated
    
#    cmap = mpl.cm.get_cmap("cool",len(peak_elements))
    bounds = np.linspace(0,len(peak_elements),len(peak_elements))       
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    

    fig, ax = plt.subplots()
    
    ax.imshow(pivot , interpolation = 'nearest',cmap=cmap)
    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    
    # Manually specify colorbar labelling after it's been generated
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(bounds)
    colorbar.set_ticklabels(peak_elements)
    
    
    # create a second axes for the colorbar
#    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
#    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    
    
    
 
    
    
    
    
    
    
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[0]):
            rect = plt.Rectangle((pivot.index[j]-.5, pivot.columns[i]-.5), 1,1, fill=False,color="k", linewidth=2)
            ax = plt.gca()
            ax.add_patch(rect)
            
            
    plt.axis('off')



    
inputFilePath = [
        r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 6\MIP Melezitose\80nm comparison\7)80nm_st95.txt',
                 r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 6\MIP Melezitose\80nm comparison\8)80nm_mel_st95.txt',
    r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 6\MIP Melezitose\80nm comparison\5)80nm_st75.txt',
    r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 6\MIP Melezitose\80nm comparison\6)80nm_mel_st75.txt',
    r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 6\MIP Melezitose\80nm comparison\1)80nm_au.txt',
    r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\own papers\paper 2\figure 6\MIP Melezitose\80nm comparison\2)80nm_mel_au.txt'
        ]
    

#inputFilePath = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
#                 r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\milk_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt'
#                 ]

neuron = 5
    
#import data in the format: elements : intensity, index : labels, columns : wavenumbers
data = ImportData(inputFilePath)

#load som
som = LoadSOM(data,neuron)

#plot the data average
PlotAvgSpec(data)

#plot som in hexag shape 
HexagonaSOMPlot(som,data)

#check the activity per most important peaks
NeuronActivationperWavePlot(som, data)

#classification
X_train, X_test, y_train, y_test = train_test_split(data.values, data.index, stratify=data.index)

print(classification_report(y_test, classify(som, X_test)))

#show the most important peaks in all the neuron activities and where they are localized
#most_relevant_feature(som)










    