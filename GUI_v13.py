import os
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import analysisFunctions_v14 as af

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




class CustomMessageBox(QMessageBox):

    def __init__(self, *__args):
        QMessageBox.__init__(self)
        self.timeout = 0
        self.autoclose = False
        self.currentTime = 0
    
    def showEvent(self, QShowEvent):
        self.currentTime = 0
        if self.autoclose:
            self.startTimer(1000)
    
    def timerEvent(self, *args, **kwargs):
        self.currentTime += 1
        if self.currentTime >= self.timeout:
            self.done(0)
    
    @staticmethod
    def showWithTimeout(timeoutSeconds, message, title, icon=QMessageBox.Information, buttons=QMessageBox.Ok):
        w = CustomMessageBox()
        w.autoclose = True
        w.timeout = timeoutSeconds
        w.setText(message)
        w.setWindowTitle(title)
        w.setIcon(icon)
        w.setStandardButtons(buttons)
        w.exec_()





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
        
        #
        self.runPCAButton.clicked.connect(self.runPCAPlot)
        self.run3DPCAButton.clicked.connect(self.run3DPCAPlot)

        
        self.AsymmetryParamBox.setText("0.05")
        self.SmoothnessParamBox.setText("1000000")
        self.MaxItersBox.setText("10")
        self.ConvThreshBox.setText("0.00001")
        
        
        	# Set variables default values for PCA
        
        self.ZscoreParamBox.setText("0")
        self.WindowminParamBox.setText("0")
        self.WindowmaxParamBox.setText("0")
        self.DistanceParam.setText("binary")

        
        # Set variables default for Peak Finder
        
        self.libraryBrowseButton.clicked.connect(self.selectLibrary)
        
    #flag to compare single to map scans
    
        
    def runPlot(self):
        if self.HelperCheckBox.isChecked():
            text = ['You selected to plot the individual average Raman data. Your data will be averaged and displayed with individual plots.\n\n']
        
        if self.FunkyCheckBox.isChecked():
            plt.xkcd()
            if self.HelperCheckBox.isChecked():
                text = text+['You chose the funky style plot; you are epic!\n\n']
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

                asymmetry_param = float(self.AsymmetryParamBox.text())
                smoothness_param = float(self.SmoothnessParamBox.text())
                max_iters = int(self.MaxItersBox.text())
                conv_thresh = float(self.ConvThreshBox.text()) 
                      
                zscore = float(self.ZscoreParamBox.text())
                
                if self.InvCheckBox.isChecked():
                    inv = 1
                else:
                    inv = 0
                
                if zscore < 0:
                    self.Zscorelow()
                    exit  
                        
                  
                wavenumber, intensity, n_points, n_raman,label, flag = af.ImportData(file)  
                

                if self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                    if flag == 0 or intensity.shape[1]<1:
                        self.single_warning_applications()
                        pass  
                    else:    
                        y_avg, y_std = af.STD(intensity)
                        baseline = af.Baseline(y_avg,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                        baseline2 = af.Baseline(y_std,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                        norm = af.Normalize(baseline,zscore,inv)
                        std_norm = baseline2/baseline.max()
                        af.ShadePlot(wavenumber,norm,std_norm,label)
                        if n == 0 and self.HelperCheckBox.isChecked():
                            text = text+['You choose the option to plot the average plot of your data points with baseline '\
                                     'removed and data normalized between 0 and 1. Since you also toggled the standard deviation box, '\
                                     'the standard deviation will be represented by an orange hue around the data points '\
                                     'the standard deviation value is mirrored up and down around the plot line.\n\n']
                        
                elif self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked():
                    if flag == 0 or intensity.shape[1]<1:
                        self.single_warning_applications()
                        pass
                    else:                        
                        y_avg, y_std = af.STD(intensity)
                        baseline = af.Baseline(y_avg,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                        baseline2 = af.Baseline(y_std,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                        af.ShadePlot(wavenumber, baseline , baseline2,label)
                        if n == 0 and self.HelperCheckBox.isChecked():
                            text = text+['You choose the option to plot the average plot of your data points with baseline '\
                                    'removed. Since you also toggled the standard deviation box, '\
                                     'the standard deviation will be represented by an orange hue around the data points '\
                                     'the standard deviation value is mirrored up and down around the plot line.\n\n']
                                    
                elif self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                    if flag == 0 or intensity.shape[1]<1:
                        self.single_warning_applications()
                        pass 
                    else:                     
                        y_avg, y_std = af.STD(intensity)
                        norm = af.Normalize(y_avg,zscore,inv)
                        std_norm = y_std/y_avg.max()
                        af.ShadePlot(wavenumber,norm,std_norm,label)
                        if n == 0 and self.HelperCheckBox.isChecked():
                            text = text+['You choose the option to plot the average plot of your data points ' \
                                     'normalized between 0 and 1. Since you also toggled the standard deviation box, '\
                                     'the standard deviation will be represented by an orange hue around the data points '\
                                     'the standard deviation value is mirrored up and down around the plot line.\n\n']
                        
                elif self.baselineCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                    if flag == 0 or intensity.shape[1]<1:
                        self.empty_warning_applications()
                        pass 
                    else:
                        
                        baseline = af.Baseline(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                        norm = af.Normalize(baseline,zscore,inv) 
                        af.IndividualPlot(wavenumber,norm,label)
                        if n == 0 and self.HelperCheckBox.isChecked():
                            text = text+['You choose the option to plot the average plot of your data points with baseline '\
                                     'removed and data normalized between 0 and 1.\n\n']
                        
                elif self.stdCheckBox.isChecked():
                    if flag == 0 or intensity.shape[1]<1:
                        self.single_warning_applications()
                        pass 
                    else:                       
                        y_avg, y_std = af.STD(intensity)
                        af.ShadePlot(wavenumber,y_avg,y_std,label)
                        if n == 0 and self.HelperCheckBox.isChecked():
                            text = text+['Since you also toggled the standard deviation box, the standard deviation will be ' \
                                     'represented by an orange hue around the data points the standard deviation is mirrored '\
                                     'up and down around the plotline. This plot doesnt seem correct, because the standard deviation '\
                                     'might appear large. This has to do with the particularities of the Raman spectra '\
                                     'which depending on your sample can induce a rather significant variation depending on the spot '\
                                     'you measured. The phenomenons that originate the more considerable difference between plots '\
                                     'could be fluorescence or backscattering. We suggest you try the standard deviation '\
                                     'with the baseline option for more appealing results :).\n\n']
                    
                elif self.baselineCheckBox.isChecked():
                    if flag == 0 or intensity.shape[1]<1:
                        self.single_warning_applications()
                        pass 
                    else:
                        baseline = af.Baseline(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)  
                        af.IndividualPlot(wavenumber, baseline, label)
                        if n == 0 and self.HelperCheckBox.isChecked():
                            text = text+['You choose the option to plot the average plot of your data points with baseline. '\
                                         'The baseline algorithm used in this work is a combination of Asymmetric-Least '\
                                         'Square fit with a Whittaker Smoothing. The parameters to control the baseline '\
                                         'are displayed in the Baseline tab. Go there and give it a try. \n\n']
                                
                    
                elif self.normalizationCheckBox.isChecked():
                    if flag == 0 or intensity.shape[1]<1:
                        self.single_warning_applications()
                        pass 
                    else:
                        norm = af.Normalize(intensity,zscore,inv)
                        af.IndividualPlot(wavenumber,norm,label)
                        if n == 0 and self.HelperCheckBox.isChecked():
                            text = text+['You choose the option to plot the average plot of your data normalized between 0 and 1.\n\n']
                        
                else:
                    if flag == 0 or intensity.shape[1]<1:
                        self.single_warning_applications()
                        pass 
                    else:
                        af.IndividualPlot(wavenumber,intensity,label)
                        if n == 0 and self.HelperCheckBox.isChecked():
                            text = text+['Ah I see you like thing au naturel, nothing against it, that is how you should see '\
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
                    'heat plot style. Where brighter colours indicate high-intensity peaks and darker colours troughs in your '\
                    'Raman data. the analogy for me is that if you were looking at different Raman spectrum from a top view. '\
                    'This technique is good to see if there are any considerable variation between your different '\
                    'dataset map. Although the heatmap has been heavily smoothed by a Savgol filter, larger impurity peaks '\
                    'can still be observed. Also, this plot is excellent to compare different experimental conditions where '\
                    'you need to track the peak lines and check which one is different. Give a try to the '\
                    'Normalised with baseline option. The heatmap does not support the standard deviation option. '\
                    'But do not worry, you can use the other options and use it there :P \n\n']
        
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
                    
                zscore = float(self.ZscoreParamBox.text())
                asymmetry_param = float(self.AsymmetryParamBox.text())
                smoothness_param = float(self.SmoothnessParamBox.text())
                max_iters = int(self.MaxItersBox.text())
                conv_thresh = float(self.ConvThreshBox.text())
                
                if self.InvCheckBox.isChecked():
                    inv = 1
                else:
                    inv = 0
                    
                if zscore<0:
                    self.Zscorelow()
                    exit  
                    
                  
                wavenumber, intensity, n_points, n_raman,label, flag = af.ImportData(file)  
                

                
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
                    norm = af.NormalizeNAVG(baseline,zscore,inv)
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
                    norm = af.NormalizeNAVG(intensity,zscore,inv)
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
                    'typical stacking form. The distance between adjacent plot has been calculated through a sophisticated method '\
                    'far superior to the usual <plus a constant> norm. Adjacent spectra were compared and the minimum difference '\
                    'between adjacent plots were added consequently. Making all plot consecutive plot intersect each other '\
                    'at a single point. If the standard deviation was selected this value was also taken into consideration '\
                    'when applying the stack plot. Please give it a go. This one is my favourite of them all! :P \n\n']
        
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

            zscore = float(self.ZscoreParamBox.text())
            asymmetry_param = float(self.AsymmetryParamBox.text())
            smoothness_param = float(self.SmoothnessParamBox.text())
            max_iters = int(self.MaxItersBox.text())
            conv_thresh = float(self.ConvThreshBox.text())
            
            if self.InvCheckBox.isChecked():
                inv = 1
            else:
                inv = 0   
                
            if zscore<0:
                self.Zscorelow()
                exit  
            
            if self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                
                if sstd.equals(empty):
                    self.single_warning_applications()
                    
                baseline = af.BaselineNAVGS(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                norm,norm2 = af.NormalizeNAVGS(baseline,zscore,inv,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                spectra = af.diff(norm,norm2)
                af.StackPlot(wavenumber, spectra, norm2,label, fig)
                
                
            elif self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                
                if sstd.equals(empty):
                    self.single_warning_applications()
                
                baseline = af.BaselineNAVGS(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)              
                baseline,baselinesstd = af.STDS(baseline)
                spectra = af.diff(baseline,baselinesstd)
                af.StackPlot(wavenumber, spectra, baselinesstd,label, fig)
                    
                            
            elif self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                
                if sstd.equals(empty):
                    self.single_warning_applications()
                
                z = af.separate_stack(z)
                norm,norm2 = af.NormalizeNAVGS(z,zscore,inv,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                spectra = af.diff(norm,norm2)
                af.StackPlot(wavenumber, spectra, norm2,label, fig)
                
            elif self.baselineCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)                
                baseline = af.BaselineNAVGS(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                norm,norm2 = af.NormalizeNAVGS(baseline,zscore,inv,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                spectra = af.diff(norm,empty)
                af.StackPlot(wavenumber, spectra, empty,label, fig)
                
            elif self.stdCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                
                if sstd.equals(empty):
                    self.single_warning_applications()

                z = af.separate_stack(z)
                z,sstd = af.STDS(z)
                spectra = af.diff(z,sstd)
                af.StackPlot(wavenumber, spectra, sstd,label, fig)
                
            elif self.baselineCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                
                baseline = af.BaselineNAVGS(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)              
                baseline,baselinesstd = af.STDS(baseline)
                spectra = af.diff(baseline,empty)
                af.StackPlot(wavenumber, spectra, empty,label, fig)
                
            elif self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                
                z = af.separate_stack(z)
                norm,norm2 = af.NormalizeNAVGS(z,zscore,inv,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                spectra = af.diff(norm,empty)
                af.StackPlot(wavenumber, spectra, empty,label, fig)
                
            else:
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                
                z = af.separate_stack(z)
                z,sstd = af.STDS(z)
                spectra = af.diff(z,empty)
                af.StackPlot(wavenumber, spectra, empty,label, fig)
                
                
        if self.HelperCheckBox.isChecked():
            self.resultsBox.setText(' '.join(text))        
#            n = n + 1


    def runWaterfallPlot(self):
        if self.HelperCheckBox.isChecked():
            text = ['You selected 3D Stack plot. The same philosophy mentioned in the stack plot is applied here but now '\
                    'with a modern 3D view twisted, which doesnt make you lose anything from sight regarding your plots. '\
                    'The sad news is that the 3D option cannot use the funky mode here due to the complexity of the patchwork under the plot line '\
                    'I guess if you want to stay funky, you have to apply it on the other buttons :( \n\n']
        
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
#            fig = plt.figure()
#            ax = fig.gca(projection='3d')
            ax = plt.axes(projection = '3d')
            plt.gcf().set_size_inches(9,9/1.618)
    #        for file in inputFilePath:
    #            n_max = len(inputFilePath)
                    
    #            wavenumber, intensity, n_points, n_raman,label = af.ImportData(file)  
                
            zscore = float(self.ZscoreParamBox.text())
            asymmetry_param = float(self.AsymmetryParamBox.text())
            smoothness_param = float(self.SmoothnessParamBox.text())
            max_iters = int(self.MaxItersBox.text())
            conv_thresh = float(self.ConvThreshBox.text())
            
            if self.InvCheckBox.isChecked():
                inv = 1
            else:
                inv = 0  
                
            if zscore<0:
                self.Zscorelow()
                exit  
        
            
            if self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                
                if sstd.equals(empty):
                    self.single_warning_applications()
              
                baseline = af.BaselineNAVGS(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                norm,norm2 = af.NormalizeNAVGS(baseline,zscore,inv,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                spectra = af.diff(norm,norm2)
                af.ThreeDPlot(wavenumber,spectra,norm2,label,ax)
                
            elif self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                
                if sstd.equals(empty):
                    self.single_warning_applications()
                    
                baseline = af.BaselineNAVGS(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)              
                baseline,baselinesstd = af.STDS(baseline)
                spectra = af.diff(baseline,baselinesstd)
                af.ThreeDPlot(wavenumber,spectra,baselinesstd,label,ax)
                            
            elif self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                
                if sstd.equals(empty):
                    self.single_warning_applications()
                
                z = af.separate_stack(z)
                norm,norm2 = af.NormalizeNAVGS(z,zscore,inv,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                spectra = af.diff(norm,norm2)
                af.ThreeDPlot(wavenumber,spectra,norm2,label,ax)
                
            elif self.baselineCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                baseline = af.BaselineNAVGS(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                norm,norm2 = af.NormalizeNAVGS(baseline,zscore,inv,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                spectra = af.diff(norm,empty)
                af.ThreeDPlot(wavenumber,spectra,empty,label,ax)
    
                
            elif self.stdCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                
                if sstd.equals(empty):
                    self.single_warning_applications()
                
                z = af.separate_stack(z)
                z,sstd = af.STDS(z)
                spectra = af.diff(z,sstd)
                af.ThreeDPlot(wavenumber,spectra,sstd,label,ax)
                
            elif self.baselineCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                baseline = af.BaselineNAVGS(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)              
                baseline,baselinesstd = af.STDS(baseline)
                spectra = af.diff(baseline,empty)
                af.ThreeDPlot(wavenumber,spectra,empty,label,ax)
                
            elif self.normalizationCheckBox.isChecked():
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                z = af.separate_stack(z)
                norm,norm2 = af.NormalizeNAVGS(z,zscore,inv,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                spectra = af.diff(norm,empty)
                af.ThreeDPlot(wavenumber,spectra,empty,label,ax)  
                
            else:  
                wavenumber,z,label,sstd = af.spec(inputFilePath)
                z = af.separate_stack(z)
                z,sstd = af.STDS(z)
                spectra = af.diff(z,empty)             
                af.ThreeDPlot(wavenumber,spectra,empty,label,ax)
                
        if self.HelperCheckBox.isChecked():
            self.resultsBox.setText(' '.join(text))             
    #            n = n + 1
    
    def runPCAPlot(self):
        if self.HelperCheckBox.isChecked():
            text = ['You selected the PCA plot. Principal Component Analysis is one of the most used statistical analysis techniques '\
                    'for Raman data analysis. It is everywhere. The main goal is to identify maximum variance vectors in your data '\
                    'and then using matrix operations to rewrite your Raman data into principal components. The PCA data analysis  '\
                    'comes equipped with a loading plot which helps you analyse the PCA differentiation. Also a '\
                    'Gaussian mixture algorithm to draw elliptical circles around your data to keep the PCA more compact. '\
                    'Here we let you choose a z parameter which will apply a Z-score onto your data Raman spectra and remove '\
                    'potential outliers based on the z value. The z value is interpreted as how many sigmas do you want to exclude. '\
                    'In my usual calculations, I use 3. But try it without any = 0 and use your judgement to analyse if it '\
                    'is a fair analysis. The other parameter is the <window> which lets you do a PCA analysis on specific '\
                    ' wavelength values, and compare locally how the different Raman spectra differentiate. Finally, we output '\
                    'the different PCA components variances, which are important to analyse how your data separates. \n\n']
        asymmetry_param = float(self.AsymmetryParamBox.text())
        smoothness_param = float(self.SmoothnessParamBox.text())
        max_iters = int(self.MaxItersBox.text())
        conv_thresh = float(self.ConvThreshBox.text())
        
        zscore = float(self.ZscoreParamBox.text())
        windowmin = float(self.WindowminParamBox.text())
        windowmax = float(self.WindowmaxParamBox.text())  
        stylestring = str(self.DistanceParam.text())  
        distance = []
        distance_k = []
        
        if windowmin > windowmax:
            self.Windowsissue() 
            exit   
        
        if zscore<0:
            self.Zscorelow()
            exit  
            
        if self.InvCheckBox.isChecked():
            inv = 1
        else:
            inv = 0   
        
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
             if self.LoadingsCheckBox.isChecked() and self.GaussianCheckBox.isChecked() and self.DistanceCheckBox.isChecked():
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                if flag == 0:
                    self.norm_warning_applications()
                    
                    
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                else:                        
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow() 
                    
                    #position.append([pos,width,height,label])
            
                    position = af.PCAPlot2D(finalDf,targets,info,flag = 1)
                    af.LoadingsPlot(wavenumber,pca,dim = 2) 
                    param, dist_substrate, dist_matrix ,dist_base = af.dist_calc(finalDf,stylestring,targets,position,dim=2)
                    distance_k = af.kdcalc(position,stylestring,dim=2)
                    distance, box_pairs,dfmelezitose,colours, box_pairs2, names2,colours2,dfdiff = af.DistanceBarPlot(param, targets,dist_substrate, dist_matrix,dist_base, stylestring)
                    af.origin_distance_plot(box_pairs, dfmelezitose,colours)
                    af.cluster_distance_plot(dfdiff,names2,stylestring,box_pairs2,colours2)
#                    af.density_plot(dfdiff,colours2,names2)
                    af.dist_k_calc(finalDf,stylestring,targets,position,colours,dim=2)
                    
             elif self.GaussianCheckBox.isChecked() and self.DistanceCheckBox.isChecked():
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                if flag == 0:
                    self.norm_warning_applications()
                    
                    
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                else:                        
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow()
                    
                    position = af.PCAPlot2D(finalDf,targets,info,flag = 1)
                    param, dist_substrate, dist_matrix ,dist_base = af.dist_calc(finalDf,stylestring,targets,position,dim=2)
                    distance_k = af.kdcalc(position,stylestring,dim=2)
                    distance, box_pairs,dfmelezitose,colours, box_pairs2, names2,colours2,dfdiff = af.DistanceBarPlot(param, targets,dist_substrate, dist_matrix,dist_base, stylestring)
                    af.origin_distance_plot(box_pairs, dfmelezitose,colours)
                    af.cluster_distance_plot(dfdiff,names2,stylestring,box_pairs2,colours2)
#                    af.density_plot(dfdiff,colours2,names2)
                    af.dist_k_calc(finalDf,stylestring,targets,position,colours,dim=2)
                    
             elif self.GaussianCheckBox.isChecked() and self.LoadingsCheckBox.isChecked():
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                if flag == 0:
                    self.norm_warning_applications()
                    
                    
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                else:                        
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.zscorelow()
                            
                    position = af.PCAPlot2D(finalDf,targets,info,flag = 1)
                    af.LoadingsPlot(wavenumber,pca,dim = 2)
                    distance_k = af.kdcalc(position,stylestring,dim=2)
                    
       
             elif self.LoadingsCheckBox.isChecked():
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                if flag == 0:
                    self.norm_warning_applications()
                    
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                else:  
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow()
                    
                    position = af.PCAPlot2D(finalDf,targets,info,flag = 0)
                    distance_k = af.kdcalc(position,stylestring,dim=2)
                    af.LoadingsPlot(wavenumber,pca,dim = 2)
                
                
             elif self.GaussianCheckBox.isChecked():
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                if flag == 0:
                    self.norm_warning_applications()
                if len(target)< 2 or len(dataset) <2: 
                    self.Zscorelow()
                else:  
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow()
                    
                    position = af.PCAPlot2D(finalDf,targets,info,flag = 1)
                    distance_k = af.kdcalc(position,stylestring,dim=2)
                    
                    print(position)
            
                                
             elif self.DistanceCheckBox.isChecked():
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                if flag == 0:
                    self.norm_warning_applications()
                    
                    
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                else:                        
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow()

                    position = af.PCAPlot2D(finalDf,targets,info,flag = 0)
                    print(position)
                    param, dist_substrate, dist_matrix ,dist_base = af.dist_calc(finalDf,stylestring,targets,position,dim=2)
                    distance_k = af.kdcalc(position,stylestring,dim=2)
                    distance, box_pairs,dfmelezitose,colours, box_pairs2, names2,colours2,dfdiff = af.DistanceBarPlot(param, targets,dist_substrate, dist_matrix,dist_base, stylestring)
                    af.origin_distance_plot(box_pairs, dfmelezitose,colours)
                    af.cluster_distance_plot(dfdiff,names2,stylestring,box_pairs2,colours2)
#                    af.density_plot(dfdiff,colours2,names2)
                    af.dist_k_calc(finalDf,stylestring,targets,position,colours,dim=2)
                
                
             else:                
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                if flag == 0:
                    self.norm_warning_applications()
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                else:  
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow()

                    position = af.PCAPlot2D(finalDf,targets,info,flag = 0)
                    distance_k = af.kdcalc(position,stylestring,dim=2)
                    print(position)
            
             if len(target)< 2 or len(dataset) <2:
                pass  
             elif len(distance_k) == 0:
                 self.resultsBox.setText(' '.join(info))
             else:
                 self.resultsBox.setText(' '.join(info+ distance_k))
#                 self.resultsBox.setText(' '.join(info+[' '.join(val) for val in distance_k]))
        
        if self.HelperCheckBox.isChecked():
            self.resultsBox.setText(' '.join(text))   
            
            
    def run3DPCAPlot(self):
        if self.HelperCheckBox.isChecked():
            text = ['You selected the 3DPCA plot. Principal Component Analysis is one of the most used statistical analysis techniques '
                    'for Raman data analysis. It is everywhere. The main goal is to identify maximum variance vectors in your data '
                    'and then using matrix operations to rewrite your Raman data into principal components. The PCA data analysis  '
                    'comes equipped with a loading plot which helps you analyse the PCA differentiation. Also a '
                    'Gaussian mixture algorithm to draw elliptical circles around your data to keep the PCA more compact. '
                    'Here we let you choose a z parameter which will apply a z-score onto your data Raman spectra and remove '
                    'potential outliers based on the z value. The z value is interpreted as how many sigmas do you want to exclude. '
                    'In my usual calculations, I use 3. But try it without any = 0 and use your judgement to analyse if it '
                    'is a fair analysis. The other parameter is the <window> which lets you do a PCA analysis on specific '
                    ' wavelength values, and compare locally how the different Raman spectra differentiate. Finally, we output '
                    'the different PCA components variances, which are important to analyse how your data separates. \n\n']
        asymmetry_param = float(self.AsymmetryParamBox.text())
        smoothness_param = float(self.SmoothnessParamBox.text())
        max_iters = int(self.MaxItersBox.text())
        conv_thresh = float(self.ConvThreshBox.text())
      
        zscore = float(self.ZscoreParamBox.text())
        windowmin = float(self.WindowminParamBox.text())
        windowmax = float(self.WindowmaxParamBox.text())  
        stylestring = str(self.DistanceParam.text())   
        distance = []
        
        if windowmin > windowmax:
            self.Windowsissue()
            exit   
        
        if zscore<0:
            self.Zscorelow()
            exit  
            
        if self.InvCheckBox.isChecked():
            inv = 1
        else:
            inv = 0  
        
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
        
        
       #flag 1 gaussian plot flag 0 no gaussian plot
       
        else:
            ax = plt.axes(projection = '3d')
            plt.gcf().set_size_inches(9,9/1.618)
            
            if self.LoadingsCheckBox.isChecked() and self.GaussianCheckBox.isChecked() and self.DistanceCheckBox.isChecked():
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                if flag == 0:
                    self.norm_warning_applications()
                
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                else:                        
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow()
                        
                    position = af.PCAPlot3D(finalDf,targets,info,ax,flag = 1)
                    af.LoadingsPlot(wavenumber,pca,dim = 3)
                    param, dist_substrate, dist_matrix ,dist_base = af.dist_calc(finalDf,stylestring,targets,position,dim=3)
                    distance_k = af.kdcalc(position,stylestring,dim=3)
                    distance, box_pairs,dfmelezitose,colours, box_pairs2, names2,colours2,dfdiff = af.DistanceBarPlot(param, targets,dist_substrate, dist_matrix,dist_base, stylestring)
                    af.origin_distance_plot(box_pairs, dfmelezitose,colours)
                    af.cluster_distance_plot(dfdiff,names2,stylestring,box_pairs2,colours2)
#                    af.density_plot(dfdiff,colours2,names2)
                    
                    
            elif self.GaussianCheckBox.isChecked() and self.DistanceCheckBox.isChecked():
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                if flag == 0:
                    self.norm_warning_applications()
                    
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                else:                        
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow()
                
                    position = af.PCAPlot3D(finalDf,targets,info,ax,flag = 1)
                    param, dist_substrate, dist_matrix ,dist_base = af.dist_calc(finalDf,stylestring,targets,position,dim=3)
                    distance_k = af.kdcalc(position,stylestring,dim=3)
                    distance, box_pairs,dfmelezitose,colours, box_pairs2, names2,colours2,dfdiff = af.DistanceBarPlot(param, targets,dist_substrate, dist_matrix,dist_base, stylestring)
                    af.origin_distance_plot(box_pairs, dfmelezitose,colours)
                    af.cluster_distance_plot(dfdiff,names2,stylestring,box_pairs2,colours2)
#                    af.density_plot(dfdiff,colours2,names2)
            
            elif self.GaussianCheckBox.isChecked() and self.LoadingsCheckBox.isChecked():
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                if flag == 0:
                    self.norm_warning_applications()
                    
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                else:                        
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow()

                    af.PCAPlot3D(finalDf,targets,info,ax,flag = 1)
                    af.LoadingsPlot(wavenumber,pca,dim = 3)
            
            elif self.LoadingsCheckBox.isChecked():
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                
                if flag == 0:
                    self.norm_warning_applications()
                    
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                else:  
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow()
                    
                    af.PCAPlot3D(finalDf,targets,info,ax,flag = 0)
                    af.LoadingsPlot(wavenumber,pca,dim = 3)
                
                
            elif self.GaussianCheckBox.isChecked():
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                
                if flag == 0:
                    self.norm_warning_applications()
                    
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                    
                else:  
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow()
                    
                    af.PCAPlot3D(finalDf,targets,info,ax,flag = 1)
            
            elif self.DistanceCheckBox.isChecked():
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                
                if flag == 0:
                    self.norm_warning_applications()
                    
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                    
                else:  
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow()
                    
                    position = af.PCAPlot3D(finalDf,targets,info,ax,flag = 0)
                    param, dist_substrate, dist_matrix ,dist_base = af.dist_calc(finalDf,stylestring,targets,position,dim=3)
                    distance_k = af.kdcalc(position,stylestring,dim=3)
                    distance, box_pairs,dfmelezitose,colours, box_pairs2, names2,colours2,dfdiff = af.DistanceBarPlot(param, targets,dist_substrate, dist_matrix,dist_base, stylestring)
                    af.origin_distance_plot(box_pairs, dfmelezitose,colours)
                    af.cluster_distance_plot(dfdiff,names2,stylestring,box_pairs2,colours2)
#                    af.density_plot(dfdiff,colours2,names2)
                
            else:                
                wavenumber,dataset,target,flag = af.sortData(inputFilePath,asymmetry_param, smoothness_param, max_iters, conv_thresh,zscore,inv)
                
                if flag == 0:
                    self.norm_warning_applications()
                
                if len(target)< 2 or len(dataset) <2:
                    self.Zscorelow()
                    
                else:  
                    finalDf,targets,info,pca,wavenumber = af.OrganizePCAData(dataset,target,wavenumber,windowmin,windowmax)
                    
                    if finalDf.shape[0]<2:
                        self.Zscorelow()
                
                    af.PCAPlot3D(finalDf,targets,info,ax,flag = 0)
            
            if len(target)< 2 or len(dataset) <2:
                pass  
            elif len(distance) == 0:
                 self.resultsBox.setText(' '.join(info))
            else:
                self.resultsBox.setText(' '.join(info+ distance_k))
#                 self.resultsBox.setText(' '.join(info+[' '.join(val) for val in distance]))

        if self.HelperCheckBox.isChecked():
            self.resultsBox.setText(' '.join(text)) 
                
        
     
    
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
        
    def Windowsissue(self):
        QMessageBox.warning(self,'Window erros',
                                  "Window min needs to be smaller than Window max, input new values",
                                  QMessageBox.Ok)  
    

    def noheatmap_warning_applications(self):
        QMessageBox.warning(self,'HeatMap does not show std',
                                  "unselect std when running HeatMap plot",
                                  QMessageBox.Ok)
        
    def nofile_warning_applications(self):
        QMessageBox.warning(self,'Files are needed to run this app',
                                  "Please select .txt or .csv files to plot.",
                                  QMessageBox.Ok)
        
    def single_warning_applications(self):
        CustomMessageBox.showWithTimeout(1, "You have a single spectrum", "You cant run Standard deviations on single spectra", icon=QMessageBox.Warning)
        
#        QMessageBox.warning(self,'You have a single spectrum',
#                                  "You cant run Standard deviations on single spectra",
#                                  QMessageBox.Ok)
        
    def empty_warning_applications(self):
#        QMessageBox.warning(self,'You have a empty spectrum',
#                                  "You change z score value.",
#                                  QMessageBox.Ok)
        CustomMessageBox.showWithTimeout(1, "You have a empty spectrum", "You change z score value.", icon=QMessageBox.Warning)
        
    def norm_warning_applications(self):
        QMessageBox.warning(self,'Normalization Error',
                                  "Your normalisation gave all Nan. Which means this spectra is a straight line at I = 0,"\
                                  " suggest remove this datapoint and redoo the Raman measurement. "\
                                  "We will pass it as a 0 value intensity line. ",
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




