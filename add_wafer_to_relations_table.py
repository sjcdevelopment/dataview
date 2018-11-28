# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:11:50 2018

@author: sswifter
"""

import photonics_upload_scripts as upload
import tkinter as tk
from tkinter import *
import tkinter.messagebox as tm


class LoginFrame(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.root = root
        self.label_username = Label(self, text="Username")
        self.label_password = Label(self, text="Password")
        self.label_wafername = Label(self, text="Wafer Name")
        self.entry_username = Entry(self)
        self.entry_password = Entry(self, show="*")
        self.entry_wafername = Entry(self)
        
        OPTIONS = ['Detector', 'EELS', 'VCSEL']
        variable = StringVar(self)
        variable.set(OPTIONS[0])
        
        self.label_devtype = Label(self, text="Device Type")
        self.entry_devtype = OptionMenu(self,variable,*OPTIONS)

        self.label_username.grid(row=0, sticky=E)
        self.label_password.grid(row=1, sticky=E)
        self.label_wafername.grid(row=2, sticky=E)
        self.label_devtype.grid(row=3, sticky=E)
        self.entry_username.grid(row=0, column=1)
        self.entry_password.grid(row=1, column=1)
        self.entry_wafername.grid(row=2, column=1)
        self.entry_devtype.grid(row=3, column=1)
        self.variable = variable.get()
        
        self.logbtn = Button(self, text="Login", command=self._login_btn_clicked)
        self.logbtn.grid(columnspan=2)

        self.pack()

    def _login_btn_clicked(self):
        # print("Clicked")
        username = self.entry_username.get()
        password = self.entry_password.get()
        wafername = self.entry_wafername.get()
        dev_type = self.variable
        upload.add_wafer_to_relations_table(wafername, username,dev_type, password=password)
        self.root.destroy()

if __name__ =='__main__':
    root = Tk()
    lf = LoginFrame(root)
    root.mainloop()