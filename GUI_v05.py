import os
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import analysisFunctions as af

# GUI:
import sys
from PyQt5 import QtCore, QtGui, uic,  QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox
import signal
import pandas as pd

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
qtCreatorFile = r"{}/stressStrainAnalyseMain_v01.ui".format(dir_path)

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
# Initialize variables
openFile = ""
testType = "Compression Test"


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    filePath = ""
    
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.fileBrowseButton.clicked.connect(self.selectFile)
        self.runPlotButton.clicked.connect(self.runPlot)
        self.runStackButton.clicked.connect(self.runStackPlot)
        self.runHeatButton.clicked.connect(self.runHeatPlot)
        self.runWaterfallButton.clicked.connect(self.runWaterfallPlot)
#        self.runStackPlotButton.clicked.connect(self.runStackedPlot)
#        self.runHeatButton.clicked.connect(self.runHeat)

		# Set variables default values
        self.AsymmetryParamBox.setText("0.05")
        self.SmoothnessParamBox.setText("1000000")
        self.MaxItersBox.setText("10")
        self.ConvThreshBox.setText("0.00001")
        
        
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
        # the selection is made from last selected to first selected. By inserting the [::-1] it forces the selected order... 
        inputFilePath = openFile[::-1]
        n = 0
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
                
            elif self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked():
                y_avg, y_std = af.STD(intensity)
                baseline = af.Baseline(y_avg,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                baseline2 = af.Baseline(y_std,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                af.ShadePlot(wavenumber, baseline , baseline2,label)
                            
            elif self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                y_avg, y_std = af.STD(intensity)
                norm = af.Normalize(y_avg)
                std_norm = y_std/y_avg.max()
                af.ShadePlot(wavenumber,norm,std_norm,label)
                
            elif self.baselineCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                baseline = af.Baseline(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                norm = af.Normalize(baseline)
                af.IndividualPlot(wavenumber,norm,label)
                
            elif self.stdCheckBox.isChecked():
                y_avg, y_std = af.STD(intensity)
                af.ShadePlot(wavenumber,y_avg,y_std,label)
                
            elif self.baselineCheckBox.isChecked():
                baseline = af.Baseline(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)  
                af.IndividualPlot(wavenumber, baseline, label)
                
            elif self.normalizationCheckBox.isChecked():
                norm = af.Normalize(intensity)
                af.IndividualPlot(wavenumber,norm,label)
                
            else:
                af.IndividualPlot(wavenumber,intensity,label)
            
            n = n + 1
            
         # The flag checks if the normalization is passed to force the values to be between 0 and 1
    def runHeatPlot(self):
        inputFilePath = openFile[::-1]
        n = 0
        for file in inputFilePath:
                
            wavenumber, intensity, n_points, n_raman,label = af.ImportData(file)  
            
            asymmetry_param = float(self.AsymmetryParamBox.text())
            smoothness_param = float(self.SmoothnessParamBox.text())
            max_iters = int(self.MaxItersBox.text())
            conv_thresh = float(self.ConvThreshBox.text())
            
            
            if  self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                self.warning_applications()
                break
                
            elif self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked():
                self.warning_applications()
                break
                            
            elif self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                self.warning_applications()
                break
                
            elif self.baselineCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                baseline = af.BaselineNAVG(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                norm = af.NormalizeNAVG(baseline)
                flag = 1
                af.PlotHeatMap(wavenumber,norm,label,flag)
                
            elif self.stdCheckBox.isChecked():
                self.warning_applications()
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

    def runStackPlot(self):
        inputFilePath = openFile[::-1]

        fig = plt.figure(figsize=(9,9/1.618))
  
        asymmetry_param = float(self.AsymmetryParamBox.text())
        smoothness_param = float(self.SmoothnessParamBox.text())
        max_iters = int(self.MaxItersBox.text())
        conv_thresh = float(self.ConvThreshBox.text())          
        
        if self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
            wavenumber,z,label,sstd = af.spec(inputFilePath)
            baseline = af.BaselineNAVG(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
            baselinesstd = af.BaselineNAVG(sstd,asymmetry_param, smoothness_param, max_iters, conv_thresh)
            spectra = af.diff(baseline,baselinesstd)
            af.StackPlot(wavenumber, spectra, baselinesstd,label, fig)
            
        elif self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked():
            wavenumber,z,label,sstd = af.spec(inputFilePath)
            baseline = af.BaselineNAVG(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
            baselinesstd = af.BaselineNAVG(sstd,asymmetry_param, smoothness_param, max_iters, conv_thresh)
            norm = af.NormalizeNAVG(baseline)
            norm2 = af.NormalizeNSTD(baseline,baselinesstd)
            spectra = af.diff(norm,norm2)
            af.StackPlot(wavenumber, spectra, norm2,label, fig)
            
                        
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
            
#            n = n + 1


    def runWaterfallPlot(self):
        inputFilePath = openFile[::-1]
#        n = 0
        ax = plt.axes(projection = '3d')
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
            spectra = af.diff(baseline,baselinesstd)
            af.ThreeDPlot(wavenumber,spectra,baselinesstd,label,ax)

            
        elif self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked():
            wavenumber,z,label,sstd = af.spec(inputFilePath)
            baseline = af.BaselineNAVG(z,asymmetry_param, smoothness_param, max_iters, conv_thresh)
            baselinesstd = af.BaselineNAVG(sstd,asymmetry_param, smoothness_param, max_iters, conv_thresh)
            norm = af.NormalizeNAVG(baseline)
            norm2 = af.NormalizeNSTD(baseline,baselinesstd)
            spectra = af.diff(norm,norm2)
            af.ThreeDPlot(wavenumber,spectra,norm2,label,ax)
                        
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
                
#            n = n + 1

            
    def selectFile(self):
        global openFile
        openFile, _filter = QtWidgets.QFileDialog.getOpenFileNames(self)
        self.selectedFileViewBox.setText(str(openFile))
        
    def warning_applications(self):
        QMessageBox.warning(self,'HeatMap does not show std',
                                  "unselect std when running HeatMap plot",
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
	window = MyApp()
	window.show()
	timer = QTimer()
	timer.start(500)  # You may change this if you wish.
	timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.
	sys.exit(app.exec_())




