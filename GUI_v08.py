import os
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import analysisFunctions as af

# GUI:
import sys
from PyQt5 import QtCore, QtGui, uic,  QtWidgets, Qt
from PyQt5.QtCore import QTimer, pyqtSignal, QEvent
from PyQt5.QtWidgets import QApplication, QMessageBox, QPushButton, QFrame, QLabel, QWidget, QVBoxLayout
import signal
import pandas as pd
from PyQt5.QtWidgets import QListView

empty = pd.DataFrame()
#def ssPlot(xList, yList):
#	# Plots stress-strain curve
#	pyplot.plot(xList, yList)
#	pyplot.xlabel("Strain (%)")
#	pyplot.ylabel("Stress (MPa)")
#	pyplot.show()

# GUI Stuff
# Code derived from:
# http://pythonforengineers.com/your-first-gui-app-with-python-and-pyqt/
dir_path = os.path.dirname(os.path.realpath(__file__))

#substitui \ por / no qtCreatorFile porque dava erro no path
qtCreatorFile = r"{}/Raman Spec UI.ui".format(dir_path)
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
# Initialize variables
openFile = ""
testType = "Compression Test"


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    filePath = ""
    
    def __init__(self):
        super().__init__()
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        # Set variables default values for plotting
        
        self.fileBrowseButton.clicked.connect(self.selectFile)
        self.runPlotButton.clicked.connect(self.runPlot)
        self.runStackButton.clicked.connect(self.runStackPlot)
        self.runHeatButton.clicked.connect(self.runHeatPlot)
        self.runWaterfallButton.clicked.connect(self.runWaterfallPlot)
        
#        self.fileBrowseButton.clicked.connect(self.selectFile)
        self.runPCAButton.clicked.connect(self.runPCAPlot)
#        self.run3DPCAButton.clicked.connect(self.run3DPCAPlot)
#        self.runLoadingsButton.clicked.connect(self.runWaterfallPlot)
        
        self.AsymmetryParamBox.setText("0.05")
        self.SmoothnessParamBox.setText("1000000")
        self.MaxItersBox.setText("10")
        self.ConvThreshBox.setText("0.00001")
        
        
        	# Set variables default values for PCA
        
        self.ZscoreParamBox.setText("0")
        self.WindowminParamBox.setText("0")
        self.WindowmaxParamBox.setText("0")

        
        # Set variables default for Peak Finder
        
        self.libraryBrowseButton.clicked.connect(self.selectLibrary)
        
        
#        self.rSquaredMinBox.setText("0.999")
#        self.numOfStepsAfterSquaredMinIsHitBox.setText("20")
#        self.elasticModBacktrackValueBox.setText("30")
#        self.yieldStressAccuracyBox.setText("100")
#        self.yieldStressFindingStepBox.setText("1")
#        self.lowElasticModulusBox.setText("1")
#        self.highElasticModCuttingRangeBox.setText("10")
#        self.plateauAnalyseSegmentLengthBox.setText("400")
#        self.plateauRegionDefiningFactorBox.setText("1.7")

    
        
    def runPlot(self):
        if self.HelperCheckBox.isChecked():
            text = ['You selected average Raman plot. Your data will be averaged and displayed with individual plots.\n\n']
        
        if self.FunkyCheckBox.isChecked():
            plt.xkcd()
            if self.HelperCheckBox.isChecked():
                text = text+['You chose the funky style plot, you are epic!\n\n']
        else:
            plt.rcdefaults()
            font = {'family' : 'Arial', 'size'   : 22}
            plt.rc('font', **font)
            if self.HelperCheckBox.isChecked():
                text = text+['Hey, if you are feeling jazzy, try the funky style ;)\n\n']
            
        # the selection is made from last selected to first selected. By inserting the [::-1] it forces the selected order... 
        inputFilePath = openFile
        
        if not inputFilePath:
            
            self.nofile_warning_applications()
        
        else:
        
            n = 0
            
            if self.HelperCheckBox.isChecked():
                text = text+['You choose '+str(len(inputFilePath))+' files, to plot. \n\n']
                
            for file in inputFilePath:
                    
                wavenumber, intensity, n_points, n_raman,label = af.ImportData(file)  
                
                asymmetry_param = float(self.AsymmetryParamBox.text())
                smoothness_param = float(self.SmoothnessParamBox.text())
                max_iters = int(self.MaxItersBox.text())
                conv_thresh = float(self.ConvThreshBox.text())
                
                if self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                    y_avg, y_std = af.STD(intensity)
                    baseline = af.Baseline(y_avg,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                    baseline2 = af.Baseline(y_std,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                    norm = af.Normalize(baseline)
                    std_norm = baseline2/baseline.max()
                    af.ShadePlot(wavenumber,norm,std_norm,label)
                    if n == 0 and self.HelperCheckBox.isChecked():
                        text = text+['You choose the option to plot the average plot of your data points with baseline '\
                                 'removed and data normalized between 0 and 1. Since you also toogled the standard deviation box, '\
                                 'the standard deviation will be represented by an orange hue around the data points '\
                                 'the standard deviation value is mirrored up and down around the plot line.\n\n']
                    
                elif self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked():
                    y_avg, y_std = af.STD(intensity)
                    baseline = af.Baseline(y_avg,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                    baseline2 = af.Baseline(y_std,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                    af.ShadePlot(wavenumber, baseline , baseline2,label)
                    if n == 0 and self.HelperCheckBox.isChecked():
                        text = text+['You choose the option to plot the average plot of your data points with baseline '\
                                'removed. Since you also toogled the standard deviation box, '\
                                 'the standard deviation will be represented by an orange hue around the data points '\
                                 'the standard deviation value is mirrored up and down around the plot line.\n\n']
                                
                elif self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                    y_avg, y_std = af.STD(intensity)
                    norm = af.Normalize(y_avg)
                    std_norm = y_std/y_avg.max()
                    af.ShadePlot(wavenumber,norm,std_norm,label)
                    if n == 0 and self.HelperCheckBox.isChecked():
                        text = text+['You choose the option to plot the average plot of your data points ' \
                                 'normalized between 0 and 1. Since you also toogled the standard deviation box, '\
                                 'the standard deviation will be represented by an orange hue around the data points '\
                                 'the standard deviation value is mirrored up and down around the plot line.\n\n']
                    
                elif self.baselineCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                    baseline = af.Baseline(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                    norm = af.Normalize(baseline)
                    af.IndividualPlot(wavenumber,norm,label)
                    if n == 0 and self.HelperCheckBox.isChecked():
                        text = text+['You choose the option to plot the average plot of your data points with baseline '\
                                 'removed and data normalized between 0 and 1.\n\n']
                    
                elif self.stdCheckBox.isChecked():
                    y_avg, y_std = af.STD(intensity)
                    af.ShadePlot(wavenumber,y_avg,y_std,label)
                    if n == 0 and self.HelperCheckBox.isChecked():
                        text = text+['Since you also toogled the standard deviation box, the standard deviation will be ' \
                                 'represented by an orange hue around the data points the standard deviation is mirrored '\
                                 'up and down around the plot line. This plot doesnt seem correct, because the standard deviation '\
                                 'might appear large. This has to do with the particularities of the Raman spectra '\
                                 'which depending on your sample can induce rather big variation depending on the spot '\
                                 'you measured. The phenomenons that originate the larger difference between plots '\
                                 'could be fluorescence or backscattering. We suggest you to try the standard deviation '\
                                 'with the baseline option for more appealing results :).\n\n']
                    
                elif self.baselineCheckBox.isChecked():
                    baseline = af.Baseline(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)  
                    af.IndividualPlot(wavenumber, baseline, label)
                    if n == 0 and self.HelperCheckBox.isChecked():
                        text = text+['You choose the option to plot the average plot of your data points with baseline. '\
                                     'The baseline algorithm used in this work is a combination of Asymetric-Least '\
                                     'Square fit with a Whittaker Smoothing. The parameters to control the baseline '\
                                     'are displayed in the Baseline tab. Go there an give it a try. \n\n']
                                
                    
                elif self.normalizationCheckBox.isChecked():
                    norm = af.Normalize(intensity)
                    af.IndividualPlot(wavenumber,norm,label)
                    if n == 0 and self.HelperCheckBox.isChecked():
                        text = text+['You choose the option to plot the average plot of your data normalized between 0 and 1.\n\n']
                    
                else:
                    af.IndividualPlot(wavenumber,intensity,label)
                    if n == 0 and self.HelperCheckBox.isChecked():
                        text = text+['Ah I see you like thing au naturel, nothing agains it, that is how you should see '\
                                 'your Raman data for the first time :). In the future try my favourite combination, '\
                                'Baseline + Normalisation + Standard deviation. \n\n']
                
                n = n + 1
#        print(text)
        if self.HelperCheckBox.isChecked():
            self.resultsBox.setText(' '.join(text))
            
         # The flag checks if the normalization is passed to force the values to be between 0 and 1
    def runHeatPlot(self):
        if self.HelperCheckBox.isChecked():
            text = ['You selected HeatMap plot. This option has the peculiarity of showing your data as individual plots in a '\
                    'heat plot style. Where brighter colours indicate high intensity peaks, and darker colours troughs in your '\
                    'Raman data. the analogy for me is that if you were looking at different Raman spectrum from a top view. '\
                    'This technique is good to see if there are any large variation between your different '\
                    'dataset map. Althouth the heatmap has been heavily smoothed by an Savgol filter, larger impurity peaks '\
                    'can still be obeserved. Also this plot is great to compare different experimental conditions where '\
                    'you just need to track the peak lines and check which one is different. Give a try to the '\
                    'Normalised with baseline option. The heatmap does not support the standard deviation option. '\
                    'But do not worrie, you can use the other options and use it there :P \n\n']
        
        if self.FunkyCheckBox.isChecked():
            plt.xkcd()
        else:
            plt.rcdefaults()
            font = {'family' : 'Arial', 'size'   : 22}
            plt.rc('font', **font)
            
        inputFilePath = openFile
        
        if not inputFilePath:
            
            self.nofile_warning_applications()
        
        else:
        
            n = 0
            for file in inputFilePath:
                    
                wavenumber, intensity, n_points, n_raman,label = af.ImportData(file)  
                
                asymmetry_param = float(self.AsymmetryParamBox.text())
                smoothness_param = float(self.SmoothnessParamBox.text())
                max_iters = int(self.MaxItersBox.text())
                conv_thresh = float(self.ConvThreshBox.text())
                
                
                if  self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                    self.noheatmap_warning_applications()
                    break
                    
                elif self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked():
                    self.noheatmap_warning_applications()
                    break
                                
                elif self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                    self.noheatmap_warning_applications()
                    break
                    
                elif self.baselineCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                    baseline = af.BaselineNAVG(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                    norm = af.NormalizeNAVG(baseline)
                    flag = 1
                    af.PlotHeatMap(wavenumber,norm,label,flag)
                    
                elif self.stdCheckBox.isChecked():
                    self.noheatmap_warning_applications()
                    break
                    
                elif self.baselineCheckBox.isChecked():
                    baseline = af.BaselineNAVG(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)  
                    flag = 0
                    af.PlotHeatMap(wavenumber, baseline, label,flag)
                    
                elif self.normalizationCheckBox.isChecked():
                    norm = af.NormalizeNAVG(intensity)
                    flag = 1
                    af.PlotHeatMap(wavenumber,norm,label,flag)
                    
                else:
                    flag = 0
                    af.PlotHeatMap(wavenumber,intensity,label,flag)
    
                n = n + 1    
                        
        if self.HelperCheckBox.isChecked():
            self.resultsBox.setText(' '.join(text))
            
            
    def runStackPlot(self):
        if self.HelperCheckBox.isChecked():
            text = ['You selected Stack plot. This option will show the different files you selected over each other in '\
                    'tipical stacking form. The distance between adjancent plot has been calculated through a sofisticated method '\
                    'far superior than the usual <plus a constant> norm. Adjacent spectra were compared and the minimum difference '\
                    'between adjacent plots were added consequently. Making all plot consecutive plot interessect each other '\
                    'at only one point. If the standard deviation was selected this values was also taken in consideration '\
                    'when applying the stack plot. Give it a go. This one is my favorite of them all! :P \n\n']
        
        if self.FunkyCheckBox.isChecked():
            plt.xkcd()
        else:
            plt.rcdefaults()
            font = {'family' : 'Arial', 'size'   : 22}
            plt.rc('font', **font)

        inputFilePath = openFile
        
        if not inputFilePath:
            
            self.nofile_warning_applications()
        
        else:

            fig = plt.figure(figsize=(9,9/1.618))
      
            asymmetry_param = float(self.AsymmetryParamBox.text())
            smoothness_param = float(self.SmoothnessParamBox.text())
            max_iters = int(self.MaxItersBox.text())
            conv_thresh = float(self.ConvThreshBox.text())          
            
            if self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                baseline = af.BaselineNAVG(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                baselinesstd = af.BaselineNAVG(sstd,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                norm = af.NormalizeNAVG(baseline)
                norm2 = af.NormalizeNSTD(baseline,baselinesstd)
                spectra = af.diff(norm,norm2)
                af.StackPlot(wavenumber, spectra, norm2,label, fig)
                
                
            elif self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                baseline = af.BaselineNAVG(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                baselinesstd = af.BaselineNAVG(sstd,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                spectra = af.diff(baseline,baselinesstd)
                af.StackPlot(wavenumber, spectra, baselinesstd,label, fig)
                
                            
            elif self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                norm = af.NormalizeNAVG(z)
                norm2 = af.NormalizeNSTD(z,sstd)
                spectra = af.diff(norm,norm2)
                af.StackPlot(wavenumber, spectra, norm2,label, fig)
                
            elif self.baselineCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                baseline = af.BaselineNAVG(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                norm = af.NormalizeNAVG(baseline)
                spectra = af.diff(norm,empty)
                af.StackPlot(wavenumber, spectra, empty,label, fig)
                
            elif self.stdCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                spectra = af.diff(z,sstd)
                af.StackPlot(wavenumber, spectra, sstd,label, fig)
                
            elif self.baselineCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                baseline = af.BaselineNAVG(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                spectra = af.diff(baseline,empty)
                af.StackPlot(wavenumber, spectra, empty,label, fig)
                
            elif self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                norm = af.NormalizeNAVG(z)
                spectra = af.diff(norm,empty)
                af.StackPlot(wavenumber, spectra, empty,label, fig)
                
            else:
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                spectra = af.diff(z,empty)
                af.StackPlot(wavenumber, spectra, empty , label, fig)
                
        if self.HelperCheckBox.isChecked():
            self.resultsBox.setText(' '.join(text))        
#            n = n + 1


    def runWaterfallPlot(self):
        if self.HelperCheckBox.isChecked():
            text = ['You selected 3D Stack plot. The same philosiphy mentioned in the stack plot is applied here but now '\
                    'with a modern 3D view twiste, which doesnt make you lose anything from sight regarding your plots. '\
                    'The sad news is that the funky mode cannot be used here due to the complexity of the patch work under the plot line '\
                    'I guess if you want to stay funky you have to apply it on the other buttons :( \n\n']
        
        if self.FunkyCheckBox.isChecked():
            plt.xkcd()
        else:
            plt.rcdefaults()
            font = {'family' : 'Arial', 'size'   : 22}
            plt.rc('font', **font)
        
        inputFilePath = openFile
        
        if not inputFilePath:
            
            self.nofile_warning_applications()
        
        else:
            
    #        n = 0
            ax = plt.axes(projection = '3d')
            plt.gcf().set_size_inches(9,9/1.618)
    #        for file in inputFilePath:
    #            n_max = len(inputFilePath)
                    
    #            wavenumber, intensity, n_points, n_raman,label = af.ImportData(file)  
                
            asymmetry_param = float(self.AsymmetryParamBox.text())
            smoothness_param = float(self.SmoothnessParamBox.text())
            max_iters = int(self.MaxItersBox.text())
            conv_thresh = float(self.ConvThreshBox.text())
            
            
            if self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                baseline = af.BaselineNAVG(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                baselinesstd = af.BaselineNAVG(sstd,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                norm = af.NormalizeNAVG(baseline)
                norm2 = af.NormalizeNSTD(baseline,baselinesstd)
                spectra = af.diff(norm,norm2)
                af.ThreeDPlot(wavenumber,spectra,norm2,label,ax)
                
            elif self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                baseline = af.BaselineNAVG(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                baselinesstd = af.BaselineNAVG(sstd,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                spectra = af.diff(baseline,baselinesstd)
                af.ThreeDPlot(wavenumber,spectra,baselinesstd,label,ax)
                            
            elif self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                norm = af.NormalizeNAVG(z)
                norm2 = af.NormalizeNSTD(z,sstd)
                spectra = af.diff(norm,norm2)
                af.ThreeDPlot(wavenumber,spectra,norm2,label,ax)
                
            elif self.baselineCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                baseline = af.BaselineNAVG(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                norm = af.NormalizeNAVG(baseline)
                spectra = af.diff(norm,empty)
                af.ThreeDPlot(wavenumber,spectra,empty,label,ax)
    
                
            elif self.stdCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                spectra = af.diff(z,sstd)
                af.ThreeDPlot(wavenumber,spectra,sstd,label,ax)
                
            elif self.baselineCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                baseline = af.BaselineNAVG(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                spectra = af.diff(baseline,empty)
                af.ThreeDPlot(wavenumber,spectra,empty,label,ax)
                
            elif self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                norm = af.NormalizeNAVG(z)
                spectra = af.diff(norm,empty)
                af.ThreeDPlot(wavenumber,spectra,empty,label,ax)
                
            else:  
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                spectra = af.diff(z,empty)              
                af.ThreeDPlot(wavenumber,spectra,empty,label,ax)
                
        if self.HelperCheckBox.isChecked():
            self.resultsBox.setText(' '.join(text))             
    #            n = n + 1
    
    def runPCAPlot(self):
        if self.HelperCheckBox.isChecked():
            text = ['You selected the PCA plot. Principal Component Analysis is one of the most used statistical analysis techniques '\
                    'for Raman data analysis. It is really everywhere. The main goal is to identify maximum varience vectors in your data '\
                    'and then using matrix operations to rewrite your raman data into principal components. The PCA data analysis  '\
                    'comes equiped with a loading plot which helps you analyse the PCA differenciation. Also a '\
                    'Gaussian mixture algorithm to draw elliptical circles around your data in order to keep the PCA more compact. '\
                    'Here we let you choose a z parameter which will apply a zscore onto your data Raman spectra and remove '\
                    'potential outliers based on the z value. The z value is interpreted as how many sigmas do you want to exclude. '\
                    'In my usual calculations I use 3. But try it without any = 0 and use your judgement to analyse if it '\
                    'is a fair analysis. The other parameter is the <window> which lets you do a PCA analysis on specific '\
                    ' wavelength values, and compare locally how the different Raman spectra differentiate. Finally we output '\
                    'the different PCA componants variences, which are important to analyse how your data separates. \n\n']
        asymmetry_param = float(self.AsymmetryParamBox.text())
        smoothness_param = float(self.SmoothnessParamBox.text())
        max_iters = int(self.MaxItersBox.text())
        conv_thresh = float(self.ConvThreshBox.text())
        
        z = float(self.ZscoreParamBox.text())
        windowmin = float(self.WindowminParamBox.text())
        windowmax = float(self.WindowmaxParamBox.text())        
        
        if self.FunkyCheckBox.isChecked():
            plt.xkcd()
        else:
            plt.rcdefaults()
            font = {'family' : 'Arial', 'size'   : 22}
            plt.rc('font', **font)
            
        # the selection is made from last selected to first selected. By inserting the [::-1] it forces the selected order... 
        inputFilePath = openFile
        
        if not inputFilePath:         
            self.nofile_warning_applications()
        
        else:
            if self.LoadingsCheckBox.isChecked() and self.GaussianCheckBox.isChecked():
                wavenumber,dataset,target = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,z,windowmin,windowmax)
                if len(info)< 2 or finalDf.shape[1] <2:
                    self.Zscorelow()
                else:
                    af.PCAPlot2D(finalDf,targets,info,flag=1)
                    af.LoadingsPlot2D(wavenumber,pca)
                
            elif self.LoadingsCheckBox.isChecked():
                wavenumber,dataset,target = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,z,windowmin,windowmax)
                if len(info)< 2 or finalDf.shape[1] <2:
                    self.Zscorelow()
                af.PCAPlot2D(finalDf,targets,info,flag=0)
                af.LoadingsPlot2D(wavenumber,pca)
                
                
            elif self.GaussianCheckBox.isChecked():
                wavenumber,dataset,target = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,z,windowmin,windowmax)
                if len(info)< 2 or finalDf.shape[1] <2:
                    self.Zscorelow()
                af.PCAPlot2D(finalDf,targets,info,flag=1)              
                
            else:                
                wavenumber,dataset,target = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,z,windowmin,windowmax)
                if len(info)< 2 or finalDf.shape[1] <2:
                    self.Zscorelow()
                af.PCAPlot2D(finalDf,targets,info,flag=0)    
            self.resultsBox.setText(' '.join(info)) 
        
        if self.HelperCheckBox.isChecked():
            self.resultsBox.setText(' '.join(text)) 
            
            
            
#            if carregar no loading:
#                
#                
#            if carregar em 3d:
#                
#                
#            else:
#                
#            for i in range(norm.shape[1]):
#                dataset.append(norm.iloc[:,i])
#                labels.append([label])
#                
#            df_pca = pd.DataFrame(dataset).reset_index(drop=True)
#            target = pd.DataFrame(labels,columns=['sample'])
#            
#            PCAComponents = 2
#            pca = PCA(n_components=PCAComponents)
#            principalComponents = pca.fit_transform(df_pca)
#            columns = ['principal component '+str(i+1) for i in range(PCAComponents)]
#            principalDf = pd.DataFrame(data = principalComponents , columns = columns)
#            finalDf = pd.concat([principalDf, target], axis = 1)
#            
#            fig = plt.figure(figsize=(9,9/1.618))
#            ax = fig.add_subplot(1,1,1)
#            for label in list(np.unique(target)):
#                indicesToKeep = finalDf['sample'] == label
#    
#                new_x = finalDf.loc[indicesToKeep, 'principal component 1']
#                new_y = finalDf.loc[indicesToKeep, 'principal component 2']
#                ax.scatter(new_x, new_y, s =100,alpha=1,label=label,marker='x')
#
#                handles, labels = plt.gca().get_legend_handles_labels()
#                by_label = OrderedDict(zip(labels, handles))
#                plt.legend(loc='best',frameon=False)
#    
#            plt.show()

                
        
    
    
    def PeakFinder(self):
        if self.FunkyCheckBox.isChecked():
            plt.xkcd()
        else:
            plt.rcdefaults()
            font = {'family' : 'Arial', 'size'   : 22}
            plt.rc('font', **font)
        
        inputLibPath = openLib
        
        if not inputLibPath:
            
            self.nofile_warning_applications()
    
    
    
    from PyQt5.QtWidgets import QListView
        
    def selectFile(self):
        global openFile
        openFile, _filter = QtWidgets.QFileDialog.getOpenFileNames(self, "File selection", "", "TXT file (*.txt) ;; CSV files (*.csv)")
#        self.selectedFileViewBox.setText(str(openFile))
        self.selectedFileViewBox.setText(' \n\n'.join((openFile)))
        
    def selectLibrary(self):
        global openLib
        openLib, _filter = QtWidgets.QFileDialog.getOpenFileNames(self, "Lib selection", "", "TXT file (*.txt) ;; CSV files (*.csv)")
        self.selectedLibraryViewBox.setText(str(openLib))
        
        
    def Zscorelow(self):
        QMessageBox.warning(self,'Z-score too low',
                                  "try increasing the Z-score between 3 and inf",
                                  QMessageBox.Ok)    
    

    def noheatmap_warning_applications(self):
        QMessageBox.warning(self,'HeatMap does not show std',
                                  "unselect std when running HeatMap plot",
                                  QMessageBox.Ok)
        
    def nofile_warning_applications(self):
        QMessageBox.warning(self,'Files are needed to run this app',
                                  "Please select .txt or .csv files to plot.",
                                  QMessageBox.Ok)

    def close_application(self):
        choice = QMessageBox.question(self, 'Close Applications?',
                                            "Are you sure you want to exit?",
                                            QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            sys.exit()
        else:
            pass
        
    def closeEvent(self,QCloseEvent):
        QCloseEvent.ignore()
        self.close_application()
        


def sigint_handler(*args):
	"""Handler for the SIGINT signal."""
	sys.stderr.write('\r')
	if QMessageBox.question(None, '', "Are you sure you want to quit?", 
                         QMessageBox.Yes | QMessageBox.No,
                         QMessageBox.No) == QMessageBox.Yes:
		QApplication.quit()

if __name__ == "__main__":
	signal.signal(signal.SIGINT, sigint_handler)
	app = QtWidgets.QApplication(sys.argv)
	app.setStyle('Fusion')
	window = MyApp()
	window.show()
	timer = QTimer()
	timer.start(500)  # You may change this if you wish.
	timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.
	sys.exit(app.exec_())




