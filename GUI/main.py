#-*-coding: utf-8-*-
import sys
from PlotGUI import *
import random
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import copy
from PyQt5.QtGui import *
from PIL import Image
import os
from PIL.ImageQt import ImageQt
from Tkinter import *
from tkFileDialog import askopenfilename
from matplotlib.figure import Figure
import pickle
import pandas as pd
from LIME_GUI import lime_libraray
from SHAP_GUI import shap_library
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class utils:
    def load_file(self):
        while(1):

            root=Tk()
            root.filename = askopenfilename(initialdir = "/home/git/F_XAI/GUI/heloc",title = "choose your file")
            file_name=root.filename
            root.quit()
            root.destroy()

            file=os.path.basename(file_name)

            if file.find('.csv') is not -1:
                break
            else:
                box=QtGui.QMessageBox(QtGui.QMessageBox.Information, _translate("Dialog","Warning", None),
                    _translate("Diglog", "choose .csv type of file",None),QtGui.QMessageBox.Ok)
                box._exec()

        if file.find('.csv') is not -1:
            t=file.find('.csv')
            file=file[0:t]
        
        if file_name.find(file) is not -1:
            m=file_name.find(file)
            
            path=file_name[0:m]

        return path , file

    def load_model(self):
        model = pickle.load(open('../Classifier/trained_model/Random_Forest.sav', 'rb'))

        return model

class GUIForm(QtWidgets.QDialog):
    def __init__(self, parent=None):

        self.Dataset='../Dataset/HELOC_allRemoved.csv'
        self.list_size=5

        QtWidgets.QDialog.__init__(self, parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.utils=utils()

        self.ui.pushButton_pre.setCheckable(True)
        self.ui.pushButton_pre.setChecked(True)
        self.ui.pushButton_pre.toggle()
        self.iniFunc()

        self.ui.Select_File.clicked.connect(self.selectfile)
        self.ui.pushButton_pre.clicked.connect(self.toggle)
        self.ui.Execution.clicked.connect(self.execution)
    def refresh(self):
        for j in range(self.list_size):
            self.L_bar_list[j].setValue(0)
            self.L_text_list[j].setText('')
            self.R_bar_list[j].setValue(0)
            self.R_text_list[j].setText('')
            self.L_bar_list[j].setTextVisible(False)
            self.R_bar_list[j].setTextVisible(False)
            self.R_value_list[j].setText('')
            self.L_value_list[j].setText('')


        self.ui.Prob1.setValue(0)
        self.ui.Prob2.setValue(0)
    def selectfile(self):

        self.refresh()


        self.path, self.file =self.utils.load_file()

    def toggle(self):
        if self.ui.pushButton_pre.isChecked():
            pixmap = QPixmap("/home/git/F_XAI/GUI/not_check.JPG")
            self.ui.LIME.setScaledContents(True)
            self.ui.LIME.setPixmap(pixmap)

            pixmap = QPixmap("/home/git/F_XAI/GUI/check.JPG")
            self.ui.SHAP.setScaledContents(True)
            self.ui.SHAP.setPixmap(pixmap)

            self.checked='SHAP'
        else:
            pixmap = QPixmap("/home/git/F_XAI/GUI/check.JPG")
            self.ui.LIME.setScaledContents(True)
            self.ui.LIME.setPixmap(pixmap)

            pixmap = QPixmap("/home/git/F_XAI/GUI/not_check.JPG")
            self.ui.SHAP.setScaledContents(True)
            self.ui.SHAP.setPixmap(pixmap)
            self.checked='LIME'


    def iniFunc(self):
        self.L_bar_list = [self.ui.X_Lbar_0, self.ui.X_Lbar_1, self.ui.X_Lbar_2, self.ui.X_Lbar_3, self.ui.X_Lbar_4]

        self.L_text_list = [self.ui.X_LText_0, self.ui.X_LText_1, self.ui.X_LText_2, self.ui.X_LText_3, self.ui.X_LText_4]

        self.L_value_list = [self.ui.X_LValue_0, self.ui.X_LValue_1, self.ui.X_LValue_2, self.ui.X_LValue_3, self.ui.X_LValue_4]

        self.R_value_list = [self.ui.X_RValue_0, self.ui.X_RValue_1, self.ui.X_RValue_2, self.ui.X_RValue_3, self.ui.X_RValue_4]

        self.R_bar_list = [self.ui.X_Rbar_0, self.ui.X_Rbar_1, self.ui.X_Rbar_2, self.ui.X_Rbar_3, self.ui.X_Rbar_4]

        self.R_text_list = [self.ui.X_RText_0, self.ui.X_RText_1, self.ui.X_RText_2, self.ui.X_RText_3, self.ui.X_RText_4 ]
        pixmap = QPixmap("/home/git/F_XAI/GUI/check.JPG")
        self.ui.LIME.setScaledContents(True)
        self.ui.LIME.setPixmap(pixmap)

        pixmap = QPixmap("/home/git/F_XAI/GUI/not_check.JPG")
        self.ui.SHAP.setScaledContents(True)
        self.ui.SHAP.setPixmap(pixmap)

        for j in range(self.list_size):
            self.L_bar_list[j].setTextVisible(False)
            self.R_bar_list[j].setTextVisible(False)
        self.checked="LIME"

    def classify_model(self):

        self.model=self.utils.load_model()
        self.file_name=self.path+self.file+'.csv'

        self.load_df=pd.read_csv(self.file_name)
        self.load_df=self.load_df.values

        prob=self.model.predict_proba(self.load_df)

        prob[0][1]=np.round(prob[0][1],2)
        prob[0][0] = np.round(prob[0][0], 2)
        self.ui.Prob1.setValue(prob[0][1]*100)
        self.ui.Prob2.setValue(prob[0][0]*100)

    def explain(self):

        if self.checked == "LIME":

            self.exp=lime_libraray(self.Dataset,self.file_name,self.model)
            feature_names ,exp=self.exp.build_for_FICO()

            if 1 in exp.local_exp.keys():
                index=1
            elif 0 in exp.local_exp.keys():
                index=0


            count_pos=0
            count_neg=0

            tmp=4
            sum=0
            for j in range(self.list_size*4):
                if exp.local_exp[index][j][1] * 100 < 0:
                    sum+= np.round(abs(exp.local_exp[index][j][1]*100))
                    if tmp == 0:
                        break
                    tmp -= 1
            tmp=0
            for j in range(self.list_size*4):

                if exp.local_exp[index][j][1]*100 >= 0:
                    sum+= np.round(abs(exp.local_exp[index][j][1]*100))
                    if tmp == 4:
                        break
                    tmp+=1
            tmp=4

            for j in range(self.list_size*4):

                if exp.local_exp[index][j][1]*100 < 0:
                    count_pos+=1
                if exp.local_exp[index][j][1]*100 >= 0:
                    count_neg+=1
            if count_pos <=tmp:
                tmp=count_pos-1
            for j in range(self.list_size*4):
                if exp.local_exp[index][j][1]*100 < 0:
                    self.R_bar_list[tmp].setTextVisible(True)
                    self.R_bar_list[tmp].setValue(np.round((np.round(abs(exp.local_exp[index][j][1]*100)))/sum*100))
                    self.R_text_list[tmp].setText(feature_names[exp.local_exp[index][j][0]])

                    self.R_value_list[tmp].setText(str(self.load_df[0][exp.local_exp[index][j][0]]))
                    if tmp == 0:
                        break
                    tmp-=1


            tmp=0
            if count_neg <self.list_size:
                tmp=self.list_size-count_neg
            for j in range(self.list_size*4):

                if exp.local_exp[index][j][1]*100 >= 0:
                    self.L_bar_list[tmp].setTextVisible(True)

                    self.L_bar_list[tmp].setValue(np.round((np.round(abs(exp.local_exp[index][j][1] * 100)))/sum*100))
                    self.L_text_list[tmp].setText(feature_names[exp.local_exp[index][j][0]])
                    self.L_value_list[tmp].setText(str(self.load_df[0][exp.local_exp[index][j][0]]))
                    if tmp == 4:
                        break
                    tmp+=1


        elif self.checked == "SHAP":
            self.exp=shap_library(self.Dataset,self.file_name,self.model)
            #neg: postivie contribution to make predicting negative class
            #pos: negative contribution to make predicting negative class
            neg,pos=self.exp.build_for_FICO()
            pos_size=pos.shape[0]
            neg_size=neg.shape[0]
            tmp = 4

            sum = 0
            for j in range(pos_size):
                    sum += int(float(pos[j][0]) * 100)
                    if tmp == 0:
                        break
                    tmp -= 1
            tmp = 0
            for j in range(neg_size):
                    sum += int(float(neg[j][0]) * 100)
                    if tmp == 4:
                        break
                    tmp += 1
            tmp=4
            if pos_size >self.list_size:
                pos_size=self.list_size
            else:
                tmp=pos_size-1

            for j in range(pos_size):
                self.R_bar_list[j].setTextVisible(False)

            for j in range(pos_size):
                self.R_bar_list[tmp].setTextVisible(True)
                self.R_value_list[tmp].setText(str(int(float(pos[j][1]))))

                self.R_bar_list[tmp].setValue(np.round(float(int(float(pos[j][0]) * 100)*100)/sum))

                self.R_text_list[tmp].setText(pos[j][2])
                if tmp==0:
                    break
                tmp-=1


            tmp=0
            if neg_size>=self.list_size:
                neg_size=self.list_size
            else:
                tmp=self.list_size-neg_size

            for j in range(neg_size):
                self.L_bar_list[j].setTextVisible(False)
            for j in range(neg_size):
                self.L_bar_list[tmp].setTextVisible(True)

                self.L_value_list[tmp].setText(str(int(float(neg[j][1])))
                                               )

                self.L_bar_list[tmp].setValue(np.round(float(int(float(neg[j][0]) * 100)*100)/sum))
                self.L_text_list[tmp].setText(neg[j][2])
                if tmp==4:
                    break
                tmp+=1

    def execution(self):
        self.refresh()
        self.classify_model()
        self.explain()

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    myapp = GUIForm()
    myapp.show()
    sys.exit(app.exec_())
