# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:31:25 2018

@author: sswifter
"""
import numpy as np
import sys
sys.path.insert(0, 'C:/Users/Flash/Desktop/solarjunction-master/solarjunction-master')
from tkinter import *
import View_Detector_or_Laser_Data as dataview
global lot
import warnings
warnings.filterwarnings("ignore")
class ListBoxChoice(object):
    def __init__(self, master=None, title=None, message=None, list=[]):
        self.master = master
        self.value = None
        self.list = list[:]
        
        self.modalPane = Toplevel(self.master)

        self.modalPane.transient(self.master)
        self.modalPane.grab_set()

        self.modalPane.bind("<Return>", self._choose)
        self.modalPane.bind("<Escape>", self._cancel)

        if title:
            self.modalPane.title(title)

        if message:
            Label(self.modalPane, text=message).pack(padx=5, pady=5)
            
        
        
        Label(self.modalPane, text='Lot Name').pack(padx=5, pady=5)
        listFrame = Frame(self.modalPane)
        listFrame.pack(side=TOP, padx=5, pady=5)
        
        scrollBar = Scrollbar(listFrame)
        scrollBar.pack(side=RIGHT, fill=Y)
        self.e = Entry(listFrame)
        self.e.pack(side=TOP)
        self.listBox = Listbox(listFrame, selectmode=EXTENDED)
        self.listBox.pack(side=LEFT, fill=Y)
        scrollBar.config(command=self.listBox.yview)
        self.listBox.config(yscrollcommand=scrollBar.set)
        self.list.sort()
        for item in self.list:
            self.listBox.insert(END, item)

        buttonFrame = Frame(self.modalPane)
        buttonFrame.pack(side=BOTTOM)

        chooseButton = Button(buttonFrame, text="Choose", command=self._choose)
        chooseButton.pack()

        cancelButton = Button(buttonFrame, text="Cancel", command=self._cancel)
        cancelButton.pack(side=RIGHT)

    def _choose(self, event=None):
        try:
            for i in range(len(self.listBox.curselection())):
                firstIndex = self.listBox.curselection()[i]
                self.value = np.append(self.value,self.list[firstIndex])
            self.lot = self.e.get()
        except IndexError:
            self.value = None
        self.modalPane.destroy()

    def _cancel(self, event=None):
        self.modalPane.destroy()
        
    def returnValue(self):
        self.master.wait_window(self.modalPane)
        return self.value,self.lot

if __name__ == '__main__':
    import random
    root = Tk()
    
    returnValue = True
    directory = dataview.get_directory('C:/Users/Flash/Desktop/Data/Detector test data', 
                              "Please select a directory with the desired wafers: ")
    list = dataview.get_all_wafernames(directory)
    while returnValue:
        returnValue = ListBoxChoice(root, "Wafer Selection", "Pick several of these crazy wafers", list).returnValue()
        
        dataview.create_prelim_report(directory, returnValue[0], lot_name=returnValue[1])
        returnValue=False