import scipy
import math
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# Gets rid of some warnings in output text
import warnings
warnings.filterwarnings("ignore")

from scipy.linalg import solveh_banded
from scipy import signal


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


inputFilePath = r'C:\Users\paulo\OneDrive - University of Birmingham\Desktop\birmingham_02\saliva\saliva data\milk\milk_785nm_static1150_3s_3acc_10%_50x_map24_toothpickv02.txt'
def ImportData(inputFilePath):
    
    df = pd.read_csv(inputFilePath, 
                          sep = '\s+',
                          header = None, 
                          skiprows = 1,
                          names = ['X','Y','Wavenumber', 'Intensity'])
    n_points = sum((df.iloc[0,1]==df.iloc[:,1][df.iloc[0,0]==df.iloc[:,0]]))
    n_raman = df.shape[0]//n_points
    wavenumber = df['Wavenumber'][0:n_points]
    
    
    dataset = []
                
        
    for i in range(df.shape[0]//n_points):
        ind_df = df['Intensity'][i*n_points:n_points*(i+1)].reset_index(drop=True)
        ind_df = ind_df.rename('spectrum '+str(i+1))
        dataset.append(ind_df)

    intensity = pd.concat(dataset,axis=1)
    
    return (wavenumber, intensity, n_points, n_raman, inputFilePath[-66:-56])


def ThreeDPlot(x,y,label,n,n_max,ax):
    if pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    
    x = x.values
    y = np.array([n]* y_avg.shape[0])
    z = y_avg
    
    ax.plot(x,  [n]* y_avg.shape[0],  z, c='k',label =str(n)+' nm', lw=4,zorder=abs(n-n_max))
    ax.plot(x,  [n]* y_avg.shape[0],  z, lw=1,zorder=abs(n-n_max))
    ax.view_init(30,-45)
    verts = [(x[i],y[i], z[i]) for i in range(z.shape[0])]+[(x.min(), y.min(),ax.axes.get_zlim()[0]),(x.max(), y.max(),ax.axes.get_zlim()[0])]
    ax.add_collection3d(Poly3DCollection([verts], alpha=0.2,edgecolor=None))
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('Gold thickness (nm)')
    ax.set_zlabel(label+' Intensity (a.u.)')
    ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
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

def incre(nmax,inputFilePath):
    imax = [0]
    for i in range(nmax):
        if i == 0:
            delta = (ImportData(inputFilePath[i])[1].mean(axis=1)-imax[i]).min()
            imax.append([ImportData(inputFilePath[i])[1].mean(axis=1)-delta])
        else:            
            delta = (ImportData(inputFilePath[i])[1].mean(axis=1)-imax[i][0]).min()
            imax.append([ImportData(inputFilePath[i])[1].mean(axis=1)-delta])
    return imax
        

def StackPlot(x,y,label,n,diff):
    if pd.DataFrame(y).shape[1]==1:
        y_avg = y
    else:
        y_avg = y.mean(axis=1)
    plt.plot(x, y_avg - diff[n+1][0].min() ,label=label,lw=2)    
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel(' Intensity (a.u.)')
    plt.legend(loc='best', prop={'size':12},frameon=False)
    plt.show()

def PlotHeatMap(x,y,label):
    y.index = x.astype(int)
    plt.figure(figsize=(9,9/1.618))
    sns.heatmap(y[::-1].T)
    plt.ylim(-1,y.shape[1]+1)
    plt.xlim(-1,y.shape[0]+1)
    b,t = plt.yticks()
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.title(label)
    
def ShadePlot(x,y1,y2,label):
    plt.figure(figsize=(9,9/1.618))
    plt.plot(x, y1,color='k',label=label,lw=2)
    plt.fill_between(x, y1- y2 , y1 + y2 , color='orange', alpha=0.5)

    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel(' Intensity (a.u.)')
    plt.legend(loc='best', prop={'size':12},frameon=False)
    
    plt.show()

def ShadeStackPlot(x,y1,y2,label,n):
    plt.plot(x, y1+y1.min()*n ,label=label,lw=2)    
    plt.fill_between(x, y1- y2 +y1.min()*n, y1 + y2 +y1.min()*n, color='orange', alpha=0.5)
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel(' Intensity (a.u.)')
    plt.legend(loc='best', prop={'size':12},frameon=False)
    plt.show()
    

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
    a = []
    for i in range(y.shape[1]):
        a.append(y.iloc[:,i].values-als_baseline(y.iloc[:,i]))
    
    y = pd.DataFrame(np.transpose(a))
    return y    
    
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
    y = pd.DataFrame(np.transpose(a))

    return y

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#def importDataFunc(inputFilePath):
#	with open(r"%s" % (inputFilePath), 'r') as dataFile:
#		xList, yList = [], []
#		line = dataFile.readline()
#		while line:
#			tempList = line.strip().split(',')
#			noString = True
#			for i, j in enumerate(tempList):
#				try:
#					tempList[i] = float(j)
#				except(ValueError):
#					noString = False
#			if noString:  # handles column headings in 1st line of csv
#				xList.append(tempList[0])
#				yList.append(tempList[1])
#                
#			line = dataFile.readline()
#		# turns data into format of xList & yList
#	maxStress = float(max(yList))
#
#	return xList, yList, maxStress
#
#
#def elasticModListGenerate(xInputlist, yInputList, elasticModStart):
#	"""removes values in data list that are less than specified value, to remove
#	error from beginning of graph when finding elastic modulus"""
#	len1 = len(xInputlist)
#	xList = [x for x in xInputlist if x > elasticModStart]
#	delta = len1 - len(xList)
#	yList = yInputList[delta: len(yInputList) + 1: 1]
#
#	return xList, yList, delta
#
#
#def findElasticMod(
#	xList, yList, elasticModFindingStep, rSquaredMin,
#	numOfStepsAfterSquaredMinIsHit, elasticModBacktrackValue):
#	""" finds elastic modulus by going along initial straight line and stopping
#	when best fit line deviates from straight line, then it goes back a few steps
#	and takes gradient"""
#	rSquaredMinHit = 0
#	# ^stores how many times r_squared value is above thresholdvalue
#	breakValue = 0  # index where best fit line no longer fits sufficiently well
#	for i in range(0, len(xList) - 1, elasticModFindingStep):
#		slope, intercept, r_value, p_value, std_error = scipy.stats.linregress(
#			xList[0: i + 1: 1], yList[0: i + 1: 1])
#		# ^applies linear regression on current slice of xlist, ylist
#		r_squared = r_value ** 2
#		if r_squared > rSquaredMin:
#			rSquaredMinHit += 1
#		if (
#			rSquaredMinHit > numOfStepsAfterSquaredMinIsHit and
#			r_squared < rSquaredMin):
#			breakValue = i
#			break
#	finalxList = xList[0: breakValue - elasticModBacktrackValue: 1]
#	finalyList = yList[0: breakValue - elasticModBacktrackValue: 1]
#	slope, intercept, r_value, p_value, std_error = scipy.stats.linregress(
#		finalxList, finalyList)
#
#	return slope, intercept, breakValue
#
#
#def makeStraightLine(
#	strainValue, deltaIndex, maxStress, inputxList, inputyList,
#	yieldStressAccuracy, yieldStressFindingStep, inputSlope, inputyIntercept):
#	"""Generates x, y coordinates that make up straight line which is used for
#	finding stress at  certain strain, using elastic mod slope based offset"""
#	# y = mx + c
#	if strainValue > max(inputxList):
#		print(
#			"""WARNING: Selected strain value is outside range of strain values
#			recorded, so following stress value will not be correct.""")
#	xIntercept = (-1 * inputyIntercept) / inputSlope
#	newLinexIntercept = xIntercept + strainValue
#	newLineyIntercept = -1 * inputSlope * newLinexIntercept
#
#	newLinexList = []
#	newLineyList = []
#	if strainValue == 0.2:
#		""" uses a lower max stress value for creating line in
#		case of 0.2 % yield stress, to speed up program"""
#		provVal = 2 * inputyList[deltaIndex]
#		if provVal < maxStress:
#			maxStress = int(2 * inputyList[deltaIndex])
#	for yValue in range(
#		math.floor(inputyList[deltaIndex]), int(maxStress * yieldStressAccuracy),
#		yieldStressFindingStep):
#		"""# produces straight line; range function can only step as an integer;
#		starting at deltaIndex means straight line starts at point where
#		'straightness' of initial line stops"""
#		yValue = yValue / yieldStressAccuracy
#		xValue = (yValue - newLineyIntercept) / inputSlope
#		newLinexList.append(xValue)
#		newLineyList.append(yValue)
#
#	return newLinexList, newLineyList
#
#
#def createCutDownLists(
#	inputSlope, lowElasticModulus, inputxList, inputyList, strainValue,
#	highElasticModCuttingRange):
#	"""Takes only the relevant section of the input curve, so finding
#	intersection point is faster"""
#	if inputSlope > lowElasticModulus:
#		if strainValue >= (max(inputxList) - highElasticModCuttingRange - 1):
#			""" prevents issues where lots of identical strain values near end mess up
#			indexing (since .index() takes lowest index)"""
#			cutDownxList = (
#				[x for x in inputxList if x > (strainValue - highElasticModCuttingRange)])
#			cutLowList = []
#			for i in inputxList:
#				if i not in cutDownxList:
#					cutLowList.append(i)
#				else:
#					break
#			numBelow = len(cutLowList)
#			startingIndex = numBelow
#			endingIndex = startingIndex + len(cutDownxList) + 1
#			cutDownyList = inputyList[startingIndex: endingIndex - 1: 1]
#
#		else:
#			cutDownxList = (
#				[x for x in inputxList if
#					x > (strainValue - highElasticModCuttingRange) and
#					x < (strainValue + highElasticModCuttingRange)])
#			cutLowList = []
#			for i in inputxList:
#				if i not in cutDownxList:
#					cutLowList.append(i)
#				else:
#					break
#			numBelow = len(cutLowList)
#			startingIndex = numBelow
#			endingIndex = startingIndex + len(cutDownxList) + 1
#			cutDownyList = inputyList[startingIndex: endingIndex - 1: 1]
#
#		return cutDownxList, cutDownyList
#
#	else:
#		return inputxList, inputyList
#
#
#def findIntersection(newLinexList, newLineyList, inputxList, inputyList):
#	"""After preprocessing is complete, goes about finding intersection of
#	straight line and orginal data curve by finding	point on striaght line
#	that is closest to a point on data curve (brute force)"""
#	mainDiffList = []
#	mainDiffListDataIndexes = []
#	for i, k in enumerate(newLinexList):
#		"""finds point on data curve that each i is closest to
#		and stores in mainDiffList"""
#		subDiffList = []
#		for j, m in enumerate(inputxList):
#			xDiff = abs(m - k)
#			yDiff = abs(inputyList[j] - newLineyList[i])
#			sumDiff = xDiff + yDiff
#			subDiffList.append(sumDiff)
#		subMinDiff = min(subDiffList)
#		subMinDiffIndex = subDiffList.index(subMinDiff)
#		# ^index in main data list is stored in mainDiffListDataIndexes
#		mainDiffList.append(subMinDiff)
#		mainDiffListDataIndexes.append(subMinDiffIndex)
#	globalMinimumDifference = min(mainDiffList)
#	globalMinimumDifferenceIndex = mainDiffList.index(globalMinimumDifference)
#	dataCurveIndexyieldPoint = (
#		mainDiffListDataIndexes[globalMinimumDifferenceIndex])
#
#	return (
#		inputxList[dataCurveIndexyieldPoint], inputyList[dataCurveIndexyieldPoint])
#
#
#def findStressAtCertainStrain(
#	inputxList, inputyList, inputSlope, inputyIntercept, strainValue, maxStress,
#	deltaIndex, yieldStressAccuracy, yieldStressFindingStep, lowElasticModulus,
#	highElasticModCuttingRange):
#	""" finds stress at certain strain (sloped up from that strain)"""
#
#	newLinexList, newLineyList = makeStraightLine(
#		strainValue, deltaIndex, maxStress, inputxList, inputyList,
#		yieldStressAccuracy, yieldStressFindingStep, inputSlope, inputyIntercept)
#
#	inputxList, inputyList = createCutDownLists(
#		inputSlope, lowElasticModulus, inputxList, inputyList, strainValue,
#		highElasticModCuttingRange)
#
#	yieldStrain, yieldStress = findIntersection(
#		newLinexList, newLineyList, inputxList, inputyList)
#
#	return yieldStress
#
#
#def findMaxStress(xInputlist, yInputList):
#	maxStress = max(yInputList)
#	maxStressIndex = yInputList.index(maxStress)
#	correspondingStrain = xInputlist[maxStressIndex]
#
#	return maxStress
#
#
#def findAreaUnderCurve(xList, yList):
#	area = np.trapz(yList, xList)
#
#	return area
#
#
#def trimData(yieldStress, plateauRegionDefiningFactor, xList, yList):
#	"""Trims data so ~only plateau region is considered,
#	to improve processing time. Specifically, it cuts off data before yield point
#	and after end of plateau region (based on multiple of yield stress)"""
#	plateauEndingStressValue = yieldStress * plateauRegionDefiningFactor
#	cutIndexStart = yList.index(yieldStress)
#	xListTrimmed = xList[cutIndexStart: len(xList): 1]
#	yListTrimmed = yList[cutIndexStart: len(yList): 1]
#	tempyList = []
#	for element in yListTrimmed:
#		if element < plateauEndingStressValue:
#			tempyList.append(element)
#		else:
#			break
#	yListTrimmed = tempyList
#	xListTrimmed = xListTrimmed[0: len(yListTrimmed): 1]
#
#	return xListTrimmed, yListTrimmed
#
#
#def generateSlopeList(xListTrimmed, yListTrimmed, plateauAnalyseSegmentLength):
#	""" stores gradient values over selected
#	interval (plateauAnalyseSegmentLength) in slopeList"""
#	slopeList = []
#	for i in range(plateauAnalyseSegmentLength, len(xListTrimmed)):
#		currentxList = xListTrimmed[i - plateauAnalyseSegmentLength: i + 1: 1]
#		currentyList = yListTrimmed[i - plateauAnalyseSegmentLength: i + 1: 1]
#		slope, intercept, r_value, p_value, std_error = scipy.stats.linregress(
#			currentxList, currentyList)
#		slopeList.append(slope)
#
#	return slopeList
#
#
#def findPeaksAndDips(
#	slopeList, plateauAnalyseSegmentLength, xListTrimmed, yListTrimmed):
#	"""Find x, y (strain, stress) cooridnates of peaks and dips"""
#	peakIndexesList = []
#	# ^stores indexes (wrt trimmed x,y lists) of points	where peaks occur
#	dipIndexesList = []
#	# ^stores indexes (wrt trimmed x,y lists) of points where dips occur
#	for i in range(0, len(slopeList) - 1):
#		if slopeList[i] < 0 and slopeList[i + 1] >= 0:  # i.e. sign change
#			dipIndexesList.append(i + int((plateauAnalyseSegmentLength / 2)))
#		elif slopeList[i] >= 0 and slopeList[i + 1] < 0:  # i.e. sign change
#			peakIndexesList.append(i + int((plateauAnalyseSegmentLength / 2)))
#		else:
#			pass
#	""" These 4 lists store x & y values for peaks and dips found,
#	ready for further analysis"""
#	peakxValues = [xListTrimmed[a] for a in peakIndexesList]
#	peakyValues = [yListTrimmed[a] for a in peakIndexesList]
#	dipxValues = [xListTrimmed[a] for a in dipIndexesList]
#	dipyValues = [yListTrimmed[a] for a in dipIndexesList]
#
#	if len(peakxValues) != len(dipyValues):
#		returnStringList.append(
#			"""ATTENTION: NUMBER OF PEAKS AND DIPS DO NOT MATCH.
#			THIS WILL RESULT IN CODE ERROR.\n""")
#
#	return peakxValues, peakyValues, dipxValues, dipyValues
#
#
#def generateReturnStringList(
#	dipxValues, dipyValues, peakxValues, peakyValues, outputDecimalPlaces):
#	returnStringList = []
#	numDips = str(len(dipxValues))
#	returnStringList.append(
#		"There are %s dips in the plateau region:\n\n" % (numDips))
#	deltaStressList = []
#	deltaStrainList = []
#
#	for i, j in enumerate(peakxValues):
#		deltaY = peakyValues[i] - dipyValues[i]
#		deltaX = dipxValues[i] - peakxValues[i]
#
#		returnStringList.append(
#			"Difference in stress between peak {} and dip {} is {} MPa\n".format(
#				str(i), str(i), str(round(deltaY, outputDecimalPlaces))))
#		deltaStressList.append(str(round(deltaY, outputDecimalPlaces)))
#		deltaStrainList.append(str(round(deltaX, outputDecimalPlaces)))
#
#	return returnStringList, numDips, deltaStressList, deltaStrainList
#
#
#def analysePlateau(
#	xList, yList, yieldStress, plateauRegionDefiningFactor,
#	plateauAnalyseSegmentLength, outputDecimalPlaces):
#	"""Analyses dips and peaks in plateau region of compression curve"""
#
#	xListTrimmed, yListTrimmed = trimData(
#		yieldStress, plateauRegionDefiningFactor, xList, yList)
#
#	slopeList = generateSlopeList(
#		xListTrimmed, yListTrimmed, plateauAnalyseSegmentLength)
#
#	peakxValues, peakyValues, dipxValues, dipyValues = findPeaksAndDips(
#		slopeList, plateauAnalyseSegmentLength, xListTrimmed, yListTrimmed)
#
#	returnStringList, numDips, delStresses, delStrains = generateReturnStringList(
#		dipxValues, dipyValues, peakxValues, peakyValues, outputDecimalPlaces)
#
#	return returnStringList, numDips, delStresses, delStrains
#
#
#def findBreakingStress(yList, outputDecimalPlaces):
#	value = round(yList[-1], outputDecimalPlaces)
#	string = "Sample breaking stress is %s MPa.\n\n" % (str(value))
#
#	return string, value