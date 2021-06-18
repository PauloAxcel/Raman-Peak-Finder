#ADDITIONAL CODES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import scipy
from scipy.linalg import solveh_banded
from scipy import stats
import warnings

import matplotlib.gridspec as gridspec
import os


font = {'family' : 'Arial',
        'size'   : 22}

plt.rc('font', **font)



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
#        label = inputFilePath.split('/')[-1][:11]
        
#        works for testing
        label = inputFilePath.split('\\')[-1][:11]

            
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


def Baseline(y,asymmetry_param, smoothness_param, max_iters, conv_thresh):
    if pd.DataFrame(y).shape[1]==1 and isinstance(y, pd.DataFrame):
        y_avg = y.iloc[:,0]
    elif pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    return y_avg - als_baseline(y_avg ,asymmetry_param, smoothness_param, max_iters, conv_thresh)




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

def smooth2(t, signal2):
    dim = signal2.shape[0]
    U, s = get_line_laplacian_eigen(dim)
    return matrix_exp_eigen(U, -s, t, signal2)


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
  def __init__(self, signal2, smoothness_param, deriv_order=1):
    self.y = signal2
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




def find_local_max(signal2):
    dim = signal2.shape[0]
    ans = np.zeros(dim)
    for i in range(dim):
        dxf = signal2[i - 1] - signal2[i]
        dxb = signal2[i - 1] - signal2[i - 2]
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


def _1Lorentzian(x, amp, cen, wid):
    return amp*wid**2/((x-cen)**2+wid**2)


    
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
 
import re


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


def peakfinder(files,smooth,asymmetry_param,smoothness_param ,max_iters,conv_thresh):
    zscore = 0
    inv = 0

    peak_vector = []
    max_value = []
    spectra = []
    
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
        err = []
        
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
                    
#                    lorentz_peak_1 = _1Lorentzian(reg, *pars_1)
                    
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
                    
#                        lorentz_peak_1 = _1Lorentzian(reg, *pars_1) 
                        
                        
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
                    
#                        lorentz_peak_1 = _1Lorentzian(reg, *pars_1) 
        
                        
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
        
#        residual_1lorentz2 = residual_1lorentz - lorentz_all2 
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


###############################################################################################################################
###############################################################################################################################





#input txt 
#lockgens = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\milk_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
#             r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt']
#input csv
#lockgens = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keymilk_785nm_.csv']

lockgens = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\1st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\2nd_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\3st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\4st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\5st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\6st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\7st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\8st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\9st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\10st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\11st_saliva_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\12st_saliva_785nm_static1150_3s_3acc_10%_50x_map26_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\13st_saliva_785nm_static1150_3s_3acc_10%_50x_map26_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\14st_saliva_785nm_static1150_3s_3acc_10%_50x_map26_toothpickv02.txt',
            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\15st_saliva_785nm_static1150_3s_3acc_10%_50x_map26_toothpickv02.txt']



#csv ALWAYS
#lockgens = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keysaliva_785n.csv',
#            r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keymilk_785nm_.csv']
#           ,
keygens = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keymilk_785nm_.csv']
#           r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\keysaliva_785n.csv']

tol = 5
#operator = 'standard'
operator = 'loadings'


lockgens = [r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\GUI for raman analysis\loadingsallmilk_785nm_.csv']



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
        stack = diff2(spec,pd.DataFrame(),stack_df.std().mean())
        
        fig,ax = plt.subplots(figsize=(9,9/1.618)) 
        ax.plot(wavenumber,stack, label = stack.columns)  
        
        for data in dataset:
            ax.fill_between(np.linspace(data[0]-tol,data[0]+tol,10),stack[data[-1]].max(),stack[data[-1]].min(),color='yellow',edgecolor='k')
            ax.text(data[0]-tol,stack[data[-1]].min(),data[1][:4],rotation=90,fontsize='xx-small')
        
        ax.legend(loc='best',frameon=False)
        ax.set_xlabel('Raman shift $(cm^{-1})$')
        ax.set_ylabel('Intensity (a. u.)')
        plt.show()
        
    if operator == 'loadings':
            
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
            plt.text(data[0]-tol,0.8*spectra.max().max(),data[1],rotation=90,fontsize='xx-small')
    #            plt.text(data[0]-tol,spectra.max().max(),(", ".join(data[2])),fontsize='xx-small')
            for Qs in spectra.columns.tolist():  
                for data2 in data[2]:
                    if Qs == data2:
                        plt.fill_between(np.linspace(data[0]-tol,data[0]+tol,10),spectra[Qs].max(),spectra[Qs].min(),color='yellow',alpha=1/3)
            
                
        plt.xlabel('Raman shift (cm$^{-1}$)')
        plt.ylabel('Loadings (a.u.)')
        labels = np.arange(xnew2.min()-xnew2.min()%50,xnew2.max()-xnew2.max()%50+51,50)
        plt.xticks(labels,rotation=90)
        plt.legend(loc='best', prop={'size':12},frameon=False)
    #    plt.yticks([])
        plt.show()
                
                








def peakSOMmatching(keygens,lockgen,tol,neuron):
    
    
    colours = []
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
    dataset = []     
    
    for i in range(neuron**2):
        colours.append(cycle[i%len(cycle)])

    for keygen in keygens:
        
        key = pd.read_csv(keygen,sep=';')     
        key = key[key['Intensity (a.u.)']>0]

        lock = pd.read_csv(lockgen[0], sep=';')
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
    spectra = diff2(stack_df,pd.DataFrame())
    plt.figure(figsize=(9,9/1.618))
    for Qs in spectra.columns.tolist():
        plt.plot(xnew2,spectra[Qs], label = Qs,lw=2,color=order['color'][order['Qs']==Qs].values[0])

    for data in dataset:
        plt.text(data[0]-tol,0.8*spectra.max().max(),data[1],rotation=90,fontsize='xx-small')
#            plt.text(data[0]-tol,spectra.max().max(),(", ".join(data[2])),fontsize='xx-small')
        for Qs in spectra.columns.tolist():
            for data2 in data[2]:
                if Qs == data2:
                    plt.fill_between(np.linspace(data[0]-tol,data[0]+tol,10),spectra[Qs].max(),spectra[Qs].min(),color='yellow',alpha=1/3)
        
            
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Loadings (a.u.)')
    labels = np.arange(xnew2.min()-xnew2.min()%50,xnew2.max()-xnew2.max()%50+51,50)
    plt.xticks(labels,rotation=90)
    plt.legend(loc='best', prop={'size':12},frameon=False)
#    plt.yticks([])
    plt.show()

       
    
    