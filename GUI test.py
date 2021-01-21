import os
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import analysisFunctions as af

# GUI:
import sys
from PyQt5 import QtCore, QtGui, uic,  QtWidgets


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
        inputFilePath = openFile
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
                

    def runStackPlot(self):
        inputFilePath = openFile
        n = 0
        nmax = len(inputFilePath)
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
                af.ShadeStackPlot(wavenumber,norm,std_norm,label,n)
                
            elif self.baselineCheckBox.isChecked() and self.stdCheckBox.isChecked():
                y_avg, y_std = af.STD(intensity)
                baseline = af.Baseline(y_avg,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                baseline2 = af.Baseline(y_std,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                af.ShadeStackPlot(wavenumber, baseline , baseline2,label,n)
                            
            elif self.stdCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                y_avg, y_std = af.STD(intensity)
                norm = af.Normalize(y_avg)
                std_norm = y_std/y_avg.max()
                af.ShadeStackPlot(wavenumber,norm,std_norm,label,n)
                
            elif self.baselineCheckBox.isChecked() and self.normalizationCheckBox.isChecked():
                baseline = af.Baseline(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                norm = af.Normalize(baseline)
                af.StackPlot(wavenumber,norm,label,n,inputFilePath)
                
            elif self.stdCheckBox.isChecked():
                y_avg, y_std = af.STD(intensity)
                af.ShadeStackPlot(wavenumber,y_avg,y_std,label,n)
                
            elif self.baselineCheckBox.isChecked():
                baseline = af.Baseline(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)  
                af.StackPlot(wavenumber, baseline, label,n,inputFilePath)
                
            elif self.normalizationCheckBox.isChecked():
                norm = af.Normalize(intensity)
                af.StackPlot(wavenumber,norm,label,n,inputFilePath)
                
            else:
                imax = af.incre(nmax,inputFilePath)
                af.StackPlot(wavenumber, intensity, label, n,imax)
                
            n = n + 1

        
    def runHeatPlot(self):
        inputFilePath = openFile
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
                baseline = af.BaselineNAVG(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)
                norm = af.NormalizeNAVG(baseline)
                af.PlotHeatMap(wavenumber,norm,label)
                
            elif self.stdCheckBox.isChecked():
                y_avg, y_std = af.STD(intensity)
                af.ShadePlot(wavenumber,y_avg,y_std,label)
                
            elif self.baselineCheckBox.isChecked():
                baseline = af.BaselineNAVG(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)  
                af.PlotHeatMap(wavenumber, baseline, label)
                
            elif self.normalizationCheckBox.isChecked():
                norm = af.NormalizeNAVG(intensity)
                af.PlotHeatMap(wavenumber,norm,label)
                
            else:
                af.PlotHeatMap(wavenumber,intensity,label)
                
            n = n + 1            


    def runWaterfallPlot(self):
        inputFilePath = openFile
        n = 0
        ax = plt.axes(projection = '3d')
        for file in inputFilePath:
            n_max = len(inputFilePath)
                
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
                af.ThreeDPlot(wavenumber,norm,label,n,n_max,ax)
                
            elif self.stdCheckBox.isChecked():
                y_avg, y_std = af.STD(intensity)
                af.ShadePlot(wavenumber,y_avg,y_std,label)
                
            elif self.baselineCheckBox.isChecked():
                baseline = af.Baseline(intensity,asymmetry_param, smoothness_param, max_iters, conv_thresh)  
                af.ThreeDPlot(wavenumber,baseline,label,n,n_max,ax)
                
            elif self.normalizationCheckBox.isChecked():
                norm = af.Normalize(intensity)
                af.ThreeDPlot(wavenumber,norm,label,n,n_max,ax)
                
            else:                
                af.ThreeDPlot(wavenumber,intensity,label,n,n_max,ax)
                
            n = n + 1




            
    def selectFile(self):
        global openFile
        openFile, _filter = QtWidgets.QFileDialog.getOpenFileNames(self)
        self.selectedFileViewBox.setText(str(openFile))
        


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())




