
# coding: utf-8

# In[241]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import pandas as pd
import tkinter as tk
import tkinter.filedialog as filedialog
import os
import fnmatch
from shutil import move
import operator
from IPython.display import display
import warnings
import re
import matplotlib.cm as cm
import convert_sig_figs
import matplotlib.backends.backend_pdf
import datetime
import warnings
warnings.filterwarnings('ignore')


def get_directory(default_dir, prompt):
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(initialdir = default_dir, title = prompt)
    return directory

class detector_data():
    def __init__(self,v,i,temp,power,name, x_coord, y_coord, wafername):
        self.Voltage = v
        self.Current = i
        self.Temperature = temp
        self.Power = power
        self.dev_name = name
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.wafername = wafername

class detector_two_point_data():
	def __init__(self, wafername, lot, dev_name, x_coord, y_coord, Imeas, Vmeas, Iset, Vset, Vcomp, Icomp, selected):
		self.wafername = wafername
		self.lot = lot
		self.dev_name = dev_name
		self.x_coord = x_coord
		self.y_coord = y_coord
		self.Imeas = Imeas
		self.Vmeas = Vmeas
		self.Iset = Iset
		self.Vset = Vset
		self.Vcomp = Vcomp
		self.Icomp = Icomp
		self.selected = selected

class laser_data():
    def __init__(self,v,i,temp,power,name,length, width, thresh_current, slope):
        self.Voltage = v
        self.Current = i
        self.Temperature = temp
        self.Power = power
        self.dev_name = name
        self.Cavity_Length = length
        self.Cavity_Width = width
        self.Threshold_Current = thresh_current
        self.Slope_Efficiency = slope
        

def retrieve_files_from_directory(directory):
    print("Parsing directory: ", directory)
    filepaths = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, '*flash*'):
            try:
                filepath = os.path.join(root, filename)
                filepaths.append(filepath)
            except:
                    print("Error parsing file path. Sorry!")

    return filepaths

def retrieve_two_point_files_from_directory(directory):
	print("Parsing directory: ", directory)
	filepaths = []
	for root, dirs, files in os.walk(directory):
		for filename in fnmatch.filter(files, '*.csv'):
			if "Flash" not in filename and "PH001" not in filename:
				try:
					filepath = os.path.join(root, filename)
					filepaths.append(filepath)
				except:
					print("Error parsing file path. Sorry!")
	return filepaths

def get_all_wafernames(filepath):
    devices = define_detector_files(filepath)
    wafernames = []
    devices = sorted(devices, key=lambda x: x.wafername)
    for i in range(0,len(devices)):
        wafernames= np.append(wafernames,devices[i].wafername)
    wafernames= list(set(wafernames))
    return wafernames

def create_detector_liv_dataframe(filepath):
    data_df = pd.read_csv(filepath, header=None,index_col=0,skiprows=43).T
    params_df = pd.read_csv(filepath, header=None,index_col=0,usecols=[0,1],error_bad_lines=False).T
    data_df.columns = data_df.columns.str.lower().str.replace(' ', '_')
    
    return data_df, params_df

def create_detector_two_point_dataframe(filepath):
    df = pd.read_csv(filepath)
    return df


def create_laser_liv_dataframe(filepath):
    data_df = pd.read_csv(filepath, header=None,index_col=0,skiprows=18).T
    params_df = pd.read_csv(filepath, header=None,index_col=0,error_bad_lines=False).T
    data_df.columns = data_df.columns.str.lower().str.replace(' ', '_')
    
    return data_df, params_df


def convert_to_floats(array):
    for i in range(len(array)):
        if type(array[i]) == str:
            array[i] = float(array[i])
    return array



def define_detector_files(filepath):
    file_list = retrieve_files_from_directory(filepath)
    devices_data = []
    for i in range(len(file_list)):
        try:
            
            data_df, params_df = create_detector_liv_dataframe(file_list[i])
            warnings.simplefilter(action='ignore')
            V = convert_to_floats(data_df['v'].values)
            I = convert_to_floats(data_df['i'].values)
            temp = float(params_df['Chuck Temperature (degrees C)'].values)
            power = float(params_df['P_In (W)'].values)
            weird_name = str(params_df['Device Name'].values)
            name = re.sub(r"\W", "", weird_name)
            x_coord = float(params_df['X Coord'])
            y_coord = float(params_df['Y Coord'])
            wafername = os.path.basename(file_list[i]).split('_')[0]
            if len(V) > 2:
	            device_data = detector_data(V,I,temp,power,name,x_coord,y_coord,wafername)
	            devices_data = np.append(devices_data, device_data)
        except:
            x=0
            #print("error with"+file_list[i])
    print(str(len(devices_data))+ " data entries loaded")
    print(str((len(file_list)-len(devices_data)))+' data entries skippped')
    return devices_data

def define_two_point_detector_data(filepath):
    file_list = retrieve_two_point_files_from_directory(filepath)
    data_points = []
    for i in range(len(file_list)):
        try:
            df= create_detector_two_point_dataframe(file_list[i])
            for index, row in df.iterrows():
            	wafername = str(row['Wafer name'])
            	lot = str(row['Lot'])
            	dev_name = str(row['Device name'])
            	x_coord = float(row['x'])
            	y_coord = float(row['y'])
            	Imeas = float(row['Imeas'])
            	Vmeas = float(row['Vmeas'])
            	Iset = float(row['Iset'])
            	Vset = float(row['Vset'])
            	Vcomp = float(row['Vcomp'])
            	Icomp = float(row['Icomp'])
            	selected = int(row['Selected'])
            	datapoint = detector_two_point_data(wafername, lot, dev_name, x_coord, y_coord, Imeas, Vmeas, Iset, Vset, Vcomp, Icomp, selected)
            	data_points = np.append(data_points, datapoint)
        except:
            x=0
    print(str(len(data_points))+ " data points ready, from " + str(len(file_list)) +' total files')
    return data_points

def define_laser_files(filepath):
    file_list = retrieve_files_from_directory(filepath)
    lasers_data = []
    for i in range(len(file_list)):
        data_df, params_df = create_laser_liv_dataframe(file_list[i])
        V = convert_to_floats(data_df['v'].values)
        I = convert_to_floats(data_df['i'].values)
        power = convert_to_floats(data_df['p'].values)
        temp = float(params_df['Temperature'].values[0])
        name = str(params_df['Laser Name'].values[0])
        length = float(params_df['Cavity Length (um)'].values[0])
        width = float(params_df['Cavity Width (um)'].values[0])
        thresh_current = float(params_df['Threshold Current (mA)'].values[0])
        slope = float(params_df['Slope Efficiency'].values[0])
        this_laser_data = laser_data(V,I,temp,power,name,length, width, thresh_current, slope)
        lasers_data = np.append(lasers_data, this_laser_data)
    return lasers_data

def filter_detector_data(devices, wafernames=['all'], device_names= ['all'], powers = ['all'], temps = ['all']):
	filtered_devices = np.copy(devices)
	if 'all' not in wafernames:
		wafername_matches = []
		for device in filtered_devices:
			if device.wafername in wafernames:
				wafername_matches = np.append(wafername_matches,device)
		filtered_devices = np.copy(wafername_matches)
	if 'all' not in device_names:
		device_matches = []
		for device in filtered_devices:
			if device.dev_name in device_names:
				device_matches = np.append(device_matches,device)
		filtered_devices = np.copy(device_matches)
	if 'all' not in powers:
		power_matches = []
		for device in filtered_devices:
			for power in powers:
				if power-.5 < device.Power < power +.5:
					power_matches = np.append(power_matches,device)
		filtered_devices = np.copy(power_matches)
	if 'all' not in temps:
		temp_matches = []
		for device in filtered_devices:
			if device.Temperature in temps:
				temp_matches = np.append(temp_matches,device)
		filtered_devices = np.copy(temp_matches)


	print('For Filters ' + str(wafernames) + 'wafers, ' + str(device_names) + 'devices, ' + str(powers) + 'powers, ' + str(temps) + 'temperatures:')
	print(str(len(filtered_devices)) + ' remaining out of ' + str(len(devices)) + ' original devices')
	return filtered_devices

def filter_detector_two_point_data(devices, wafernames=['all'], device_names= ['all'], only_best= 'no'):
	filtered_devices = np.copy(devices)
	if 'all' not in wafernames:
		wafername_matches = []
		for device in filtered_devices:
			if device.wafername in wafernames:
				wafername_matches = np.append(wafername_matches,device)
		filtered_devices = np.copy(wafername_matches)
	if 'all' not in device_names:
		device_matches = []
		for device in filtered_devices:
			if device.dev_name in device_names:
				device_matches = np.append(device_matches,device)
		filtered_devices = np.copy(device_matches)
	if only_best == 'yes':
		best_devices = []
		for device in filtered_devices:
			if device.selected == 1:
				best_devices = np.append(best_devices,device)
		filtered_devices = np.copy(best_devices)
	
	print('For Filters ' + str(wafernames) + 'wafers, ' + str(device_names) + 'devices, only best devices:' + str(only_best))
	print(str(len(filtered_devices)) + ' remaining out of ' + str(len(devices)) + ' original points')

	return filtered_devices

def select_best_device(devices,voltage=-1):
	best_dark_current = 1
	best_device = []
	for device in devices:
		for i in range(len(device.Voltage)):
			if voltage-.3 <device.Voltage[i]<voltage+.3:
				if abs(device.Current[i]) < best_dark_current:
					best_dark_current = device.Current[i]
					best_device = device
	return best_device
def best_device_sorting(devices, sort_by='blank' , duplicate_devices = 'most_recent', voltage = -1):

	best_devices = []

	if sort_by == 'wafername':
		devices = sorted(devices, key=lambda x: x.wafername)
		wafernames = []
		for i in range(0,len(devices)):
			wafernames= np.append(wafernames,devices[i].wafername)
		wafernames= list(set(wafernames))
		
		for wafername in wafernames:
			wafer_devices = []
			for device in devices:
				if device.wafername == wafername:
					wafer_devices = np.append(wafer_devices, device)
			best_device = select_best_device(wafer_devices, voltage = voltage)
			best_devices = np.append(best_devices,best_device)
	return best_devices



def plot_detector_data_by_power(devices,graph_type='linear',wafername='wafer', device_name = 'various devices', temp='0 C'):
    colors = ['r', 'b', 'g', 'k', 'm']
    patches = ['r', 'b', 'g', 'k', 'm']
    powers = []
    devices = sorted(devices, key=lambda x: x.Power)
    for i in range(0,len(devices)):
        powers= np.append(powers,devices[i].Power)
    powers= list(set(powers))
    if graph_type == 'linear':
        for i in range(0,len(devices)):
            for j in range(0,len(powers)):
                if devices[i].Power == powers[j]:
                    if devices[i].Power != devices[i-1].Power or i==0:
                        plt.plot(devices[i].Voltage,devices[i].Current, color=colors[j], label= str(powers[j]) + ' mW')
                    else:
                        plt.plot(devices[i].Voltage,devices[i].Current, color=colors[j])
    if graph_type == 'log':
        for i in range(0,len(devices)):
            for j in range(0,len(powers)):
                if devices[i].Power == powers[j]:
                    if devices[i].Power != devices[i-1].Power or i==0:
                        plt.semilogy(devices[i].Voltage,np.abs(devices[i].Current), color=colors[j], label= str(powers[j]) + ' mW')
                    else:
                        plt.semilogy(devices[i].Voltage,np.abs(devices[i].Current), color=colors[j])

    
    plt.xlabel('Voltage (V)')
    plt.title(device_name + ' on '+wafername + " @ " + temp)
    plt.ylabel('Current (I)')
    plt.grid()
    plt.legend()
    plt.show()

def plot_detector_two_point_data_by_wafer(datapoints, best_devices='no', dev_name = "various devices", lot = 'lot',show=True):
	wafernames = []
	iset = []
	vset = []
	imeas = []
	vmeas = []
	colors = cm.hsv(np.linspace(0, 1, 10))
	datapoints = sorted(datapoints, key=lambda x: x.wafername)
	for i in range(0,len(datapoints)):
		wafernames= np.append(wafernames,datapoints[i].wafername)
		iset= np.append(iset,datapoints[i].Iset)
		vset= np.append(vset,datapoints[i].Vset)
		imeas = np.append(imeas,datapoints[i].Imeas)
		vmeas = np.append(vmeas,datapoints[i].Vmeas)
	wafernames= list(set(wafernames))
	fig = plt.figure(figsize=(19.2,10.8), dpi=100)
	plt.subplot(211)
	plt.title("Two Point Data for Lot: " + lot + " Device: " + dev_name + ". Best Devices: " + best_devices)
	for i in range(0,len(datapoints)):
		for j in range(0,len(wafernames)):
			if datapoints[i].wafername == wafernames[j]:
				if datapoints[i].wafername != datapoints[i-1].wafername or i==0:
					plt.plot(datapoints[i].Vmeas,datapoints[i].Imeas, color=colors[j], marker = 'o', label= str(wafernames[j]))
				else:
					plt.plot(datapoints[i].Vmeas,datapoints[i].Imeas, color=colors[j], marker = 'o')
	
	
	plt.xlabel("Turn on Voltage (V) @ " + str(np.mean(iset))+ " mA +/- " + str(np.std(iset)) + " mA")
	plt.ylabel("Dark current (mA) @ " + str(np.mean(vset)) + " V +/- " + str(np.std(vset)) + " V")
	plt.legend()
	plt.grid()
	plt.subplot(223)
	plt.hist(imeas, bins = 50)
	plt.title("Dark Currents")
	plt.xlabel("Dark Current (mA) @ " + str(np.mean(vset)) + " V +/- " + str(np.std(vset)) + " V")
	plt.ylabel("Counts")
	plt.subplot(224)
	plt.hist(vmeas, bins = 50)
	plt.title("Turn on Voltages")
	plt.xlabel("Turn on Voltage (V)  @ " +  str(np.mean(iset))+ " mA +/- " + str(np.std(iset)) + " mA")
	plt.ylabel("Counts")
	#mng = plt.get_current_fig_manager()
	#mng.window.showMaximized()
	if show==True:
		plt.show()

def map_best_detector_two_point_data_by_wafer(datapoints, dev_name = "various devices", lot = 'lot',show=True):
	wafernames = []
	colors = cm.hsv(np.linspace(0, 1, 10))
	datapoints = sorted(datapoints, key=lambda x: x.wafername)
	for i in range(0,len(datapoints)):
		wafernames= np.append(wafernames,datapoints[i].wafername)
	wafernames= list(set(wafernames))
	print(len(wafernames))
	fig=plt.figure(figsize=(19.2,10.8), dpi=100)
	    
	fig.suptitle("Best Device Map for Lot: " + lot + " Device: " + dev_name)
	for i in range(0,len(datapoints)):
		for j in range(0,len(wafernames)):
			plt.subplot(int(len(wafernames)/3)+1,3,j+1)
			wafer_outline = plt.Circle((5000, 30000), 50000, color='blue', fill=False)
			plt.xlabel("X Coordinate")
			plt.ylabel("Y Coordinate")
			plt.gcf().gca().add_artist(wafer_outline)
			plt.ylim(-25000,85000)
			plt.xlim(-50000,60000)
			plt.legend()
			if datapoints[i].wafername == wafernames[j]:
				if datapoints[i].selected == 1:
					if datapoints[i].wafername != datapoints[i-1].wafername or i==0:
						plt.plot(datapoints[i].x_coord,datapoints[i].y_coord, color=colors[j], marker = 'o', label= str(wafernames[j]))
					else:
						plt.plot(datapoints[i].x_coord,datapoints[i].y_coord, color=colors[j], marker = 'o')
	if show==True:
		plt.show()

def find_shorts_open_circuits(datapoints, dev_name = "various devices", lot = 'lot'):
	wafernames = []
	colors = cm.hsv(np.linspace(0, 1, 10))
	datapoints = sorted(datapoints, key=lambda x: x.wafername)
	for i in range(0,len(datapoints)):
		wafernames= np.append(wafernames,datapoints[i].wafername)
	wafernames= list(set(wafernames))
	fig, ax = plt.subplots()
	fig.suptitle("Potential Shorts/Open Circuits for: " + lot + " Device: " + dev_name)
	for i in range(0,len(datapoints)):
		for j in range(0,len(wafernames)):
			plt.subplot(int(len(wafernames)/3)+1,3,j+1)
			wafer_outline = plt.Circle((5000, 30000), 50000, color='blue', fill=False)
			plt.xlabel("X Coordinate")
			plt.ylabel("Y Coordinate")

			plt.ylim(-25000,85000)
			plt.xlim(-50000,60000)
			
			if datapoints[i].Vcomp-.01 <= datapoints[i].Vmeas <= datapoints[i].Vcomp+.01:
				if datapoints[i].wafername == wafernames[j]:
					if datapoints[i].wafername != datapoints[i-1].wafername or i==0:
						plt.plot(datapoints[i].x_coord,datapoints[i].y_coord, color=colors[j], marker = 'o', label= str(wafernames[j]))
						plt.legend()
						plt.gcf().gca().add_artist(wafer_outline)
					else:
						plt.plot(datapoints[i].x_coord,datapoints[i].y_coord, color=colors[j], marker = 'o')
			elif datapoints[i].Icomp-.00001 <= datapoints[i].Imeas <= datapoints[i].Icomp+.00001:
				if datapoints[i].wafername == wafernames[j]:
					if datapoints[i].wafername != datapoints[i-1].wafername or i==0:
						plt.plot(datapoints[i].x_coord,datapoints[i].y_coord, color=colors[j], marker = 'x', label= str(wafernames[j]))
						plt.legend()
						plt.gcf().gca().add_artist(wafer_outline)
					else:
						plt.plot(datapoints[i].x_coord,datapoints[i].y_coord, color=colors[j], marker = 'x')

def plot_detector_data_by_wafer(devices,graph_type='linear', device_name = 'various devices', temp='0 C', power = '0',show=True):
    colors = cm.hsv(np.linspace(0, 1, 10))
    patches = ['r', 'b', 'g', 'k', 'm']
    wafernames = []
    devices = sorted(devices, key=lambda x: x.wafername)
    for i in range(0,len(devices)):
        wafernames= np.append(wafernames,devices[i].wafername)
    wafernames= list(set(wafernames))
    if graph_type == 'linear':
        for i in range(0,len(devices)):
            for j in range(0,len(wafernames)):
                if devices[i].wafername == wafernames[j]:
                    if devices[i].wafername != devices[i-1].wafernames or i==0:
                        plt.plot(devices[i].Voltage,devices[i].Current, color=colors[j], label= str(wafernames[j]))
                    else:
                        plt.plot(devices[i].Voltage,devices[i].Current, color=colors[j])
    if graph_type == 'log':
        for i in range(0,len(devices)):
            for j in range(0,len(wafernames)):
                if devices[i].wafername == wafernames[j]:
                    if devices[i].wafername != devices[i-1].wafername or i==0:
                        plt.semilogy(devices[i].Voltage,np.abs(devices[i].Current), color=colors[j], label= str(wafernames[j]))
                    else:
                        plt.semilogy(devices[i].Voltage,np.abs(devices[i].Current), color=colors[j])

    
    plt.xlabel('Voltage (V)')
    plt.title(device_name + ' @ '+ str(power) + " mW @ " + temp)
    plt.ylabel('Current (I)')
    plt.grid()
    plt.legend()
    if show==True:
        plt.show()


def plot_detector_data_by_temp(devices,graph_type='linear',wafername='wafer',power = '0'):
    colors = cm.hsv(np.linspace(0, 1, 10))
    patches = ['r', 'b', 'g', 'k', 'm']
    temps = []
    devices = sorted(devices, key=lambda x: x.Temperature)
    for i in range(0,len(devices)):
        temps= np.append(temps,devices[i].Temperature)
    temps= list(set(temps))
    if graph_type == 'linear':
        for i in range(0,len(devices)):
            for j in range(0,len(temps)):
                if devices[i].Temperature == temps[j]:
                    if devices[i].Temperature != devices[i-1].Temperature or i==0:
                        plt.plot(devices[i].Voltage,devices[i].Current, color=colors[j], label= str(temps[j]) + ' C')
                    else:
                        plt.plot(devices[i].Voltage,devices[i].Current, color=colors[j])
    if graph_type == 'log':
        for i in range(0,len(devices)):
            for j in range(0,len(temps)):
                if devices[i].Temperature == temps[j]:
                    if devices[i].Temperature != devices[i-1].Temperature or i==0:
                        plt.semilogy(devices[i].Voltage,np.abs(devices[i].Current), color=colors[j], label= str(temps[j]) + ' C')
                    else:
                        plt.semilogy(devices[i].Voltage,np.abs(devices[i].Current), color=colors[j])

    
    plt.xlabel('Voltage (V)')
    plt.title(devices[0].dev_name + ' on '+wafername + " @ " + power + " mW")
    plt.ylabel('Current (I)')
    plt.grid()
    plt.legend()
    plt.show()



def plot_detector_data_by_name(devices,graph_type='linear',wafername='wafer',temp = '0 C', power='0 mW'):
    colors = cm.hsv(np.linspace(0, 1, 10))
    patches = ['r', 'b', 'g', 'k', 'm']
    names = []
    devices = sorted(devices, key=lambda x: x.dev_name)
    for i in range(0,len(devices)):
        names= np.append(names,devices[i].dev_name)
    names= list(set(names))
    if graph_type == 'linear':
        for i in range(0,len(devices)):
            for j in range(0,len(names)):
                if devices[i].dev_name == names[j]:
                    if devices[i].dev_name != devices[i-1].dev_name or i==0:
                        plt.plot(devices[i].Voltage,devices[i].Current, color=colors[j], label= str(names[j]))
                    else:
                        plt.plot(devices[i].Voltage,devices[i].Current, color=colors[j])
    if graph_type == 'log':
        for i in range(0,len(devices)):
            for j in range(0,len(names)):
                if devices[i].dev_name == names[j]:
                    if devices[i].dev_name != devices[i-1].dev_name or i==0:
                        plt.semilogy(devices[i].Voltage,np.abs(devices[i].Current), color=colors[j], label= str(names[j]))
                    else:
                        plt.semilogy(devices[i].Voltage,np.abs(devices[i].Current), color=colors[j])

    
    plt.xlabel('Voltage (V)')
    plt.title(wafername + " @ " + temp+ ' and ' + power)
    plt.ylabel('Current (I)')
    plt.grid()
    plt.legend()
    plt.show()

def create_prelim_report(filepath, wafernames, lot_name = "LOT"):

    all_devices = define_detector_files(filepath)
    lot_data = filter_detector_data(all_devices,wafernames)

    # Define Dark IV Data
    Open_Diode_100_by_wafer_dark = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Open_Diode_100'], powers = [0], temps = ['all'])
    Open_Diode_300_by_wafer_dark = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Open_Diode_300'], powers = [0], temps = ['all'])
    Open_Diode_5_100_by_wafer_dark = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Open_Diode_5_100'], powers = [0], temps = ['all'])
    Open_Diode_5_300_by_wafer_dark = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Open_Diode_5_300'], powers = [0], temps = ['all'])
    Full_Diode_100_by_wafer_dark = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Full_Diode_100'], powers = [0], temps = ['all'])
    Full_Diode_300_by_wafer_dark = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Full_Diode_300'], powers = [0], temps = ['all'])

    #Plot and save dark IV data
	
    fig1= plt.figure(1,figsize=(19.2,10.8), dpi=100)
    fig1.suptitle('Dark IV for ' + lot_name, fontsize=16)
    plt.subplots_adjust(wspace = .2, hspace=.4)
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

    matplotlib.rc('font', **font)
	
    ax1 = plt.subplot(321)
    plot_detector_data_by_wafer(Open_Diode_100_by_wafer_dark,graph_type='log', device_name = 'Open_Diode_100', temp='25 C', power = '0',show=False)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax2 = plt.subplot(322)
    plot_detector_data_by_wafer(Open_Diode_300_by_wafer_dark,graph_type='log', device_name = 'Open_Diode_300', temp='25 C', power = '0',show=False)
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax3 = plt.subplot(323)
    plot_detector_data_by_wafer(Open_Diode_5_100_by_wafer_dark,graph_type='log', device_name = 'Open_Diode_5_100', temp='25 C', power = '0',show=False)
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax4= plt.subplot(324)
    plot_detector_data_by_wafer(Open_Diode_5_300_by_wafer_dark,graph_type='log', device_name = 'Open_Diode_5_300', temp='25 C', power = '0',show=False)
    box = ax4.get_position()
    ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax5 = plt.subplot(325)
    plot_detector_data_by_wafer(Full_Diode_100_by_wafer_dark,graph_type='log', device_name = 'Full_Diode_100', temp='25 C', power = '0',show=False)
    box = ax5.get_position()
    ax5.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax6 = plt.subplot(326)
    plot_detector_data_by_wafer(Full_Diode_300_by_wafer_dark,graph_type='log', device_name = 'Full_Diode_300', temp='25 C', power = '0',show=False)
    box = ax6.get_position()
    ax6.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

	#mng = plt.get_current_fig_manager()
	#mng.window.showMaximized()
	# Define and plot 1mW IV Data
    Open_Diode_100_by_wafer_1mW = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Open_Diode_100'], powers = [1], temps = ['all'])
    Open_Diode_300_by_wafer_1mW = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Open_Diode_300'], powers = [1], temps = ['all'])
    Open_Diode_5_100_by_wafer_1mW = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Open_Diode_5_100'], powers = [1], temps = ['all'])
    Open_Diode_5_300_by_wafer_1mW = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Open_Diode_5_300'], powers = [1], temps = ['all'])

    fig2 = plt.figure(2,figsize=(19.2,10.8), dpi=100)
    plt.subplots_adjust(wspace = .3, hspace=.4)
    fig2.suptitle('Illuminated (1mW) IV for ' + lot_name, fontsize=16)

    ax1 = plt.subplot(221)
    plot_detector_data_by_wafer(Open_Diode_100_by_wafer_1mW,graph_type='log', device_name = 'Open_Diode_100', temp='25 C', power = '1',show=False)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax2 = plt.subplot(222)
    plot_detector_data_by_wafer(Open_Diode_300_by_wafer_1mW,graph_type='log', device_name = 'Open_Diode_300', temp='25 C', power = '1',show=False)
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax3 = plt.subplot(223)
    plot_detector_data_by_wafer(Open_Diode_5_100_by_wafer_1mW,graph_type='log', device_name = 'Open_Diode_5_100', temp='25 C', power = '1',show=False)
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax4 = plt.subplot(224)
    plot_detector_data_by_wafer(Open_Diode_5_300_by_wafer_1mW,graph_type='log', device_name = 'Open_Diode_5_300', temp='25 C', power = '1',show=False)
    box = ax4.get_position()
    ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)


	#mng = plt.get_current_fig_manager()
	#mng.window.showMaximized()

	# Define 5mW IV Data
    Open_Diode_100_by_wafer_5mW = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Open_Diode_100'], powers = [5], temps = ['all'])
    Open_Diode_300_by_wafer_5mW = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Open_Diode_300'], powers = [5], temps = ['all'])
    Open_Diode_5_100_by_wafer_5mW = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Open_Diode_5_100'], powers = [5], temps = ['all'])
    Open_Diode_5_300_by_wafer_5mW = filter_detector_data(lot_data, wafernames=['all'], device_names= ['Open_Diode_5_300'], powers = [5], temps = ['all'])

    fig3 = plt.figure(3,figsize=(19.2,10.8), dpi=100)
    plt.subplots_adjust(wspace = .3, hspace=.4)
    fig3.suptitle('Illuminated (5mW) IV for ' + lot_name, fontsize=16)
	
    ax1 = plt.subplot(221)
    plot_detector_data_by_wafer(Open_Diode_100_by_wafer_5mW,graph_type='log', device_name = 'Open_Diode_100', temp='25 C', power = '5',show=False)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax2 = plt.subplot(222)
    plot_detector_data_by_wafer(Open_Diode_300_by_wafer_5mW,graph_type='log', device_name = 'Open_Diode_300', temp='25 C', power = '5',show=False)
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax3 = plt.subplot(223)
    plot_detector_data_by_wafer(Open_Diode_5_100_by_wafer_5mW,graph_type='log', device_name = 'Open_Diode_5_100', temp='25 C', power = '5',show=False)
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    ax4 = plt.subplot(224)
    plot_detector_data_by_wafer(Open_Diode_5_300_by_wafer_5mW,graph_type='log', device_name = 'Open_Diode_5_300', temp='25 C', power = '5',show=False)
    box = ax4.get_position()
    ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

	#mng = plt.get_current_fig_manager()
	#mng.window.showMaximized()

    datapoints = define_two_point_detector_data(filepath)
    good_dev = filter_detector_two_point_data(datapoints, wafernames, device_names= ['Open_Diode_100'], only_best= 'yes')
    plot_detector_two_point_data_by_wafer(good_dev, best_devices='yes', dev_name = "Open_Diode_100", lot = 'lot',show=False)

    map_best_detector_two_point_data_by_wafer(good_dev, dev_name = "Open_Diode_100", lot = lot_name,show=False)
    timestamp = str(datetime.datetime.now()).split('.')[0].replace(":","")
    filename = "C:/Users/Flash/Desktop/Reports/" + str(lot_name) + " - " +str(timestamp) + ".pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    for fig in range(1, plt.gcf().number + 1): 
        
        pdf.savefig( fig )
    pdf.close()
	#mng.window.showMaximized()

def plot_laser_data_by_temp(devices, current = 'absolute', wafername='wafer', plot = 'LI'):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    patches = ['r', 'b', 'g', 'k', 'm']
    temps = []
    devices = sorted(devices, key=lambda x: x.Temperature)
    for i in range(0,len(devices)):
        temps= np.append(temps,devices[i].Temperature)
    temps= list(set(temps))
    current_label = 'Current (mA)'
    if current == 'density':
        for i in range(0,len(devices)):
            area = devices[i].Cavity_Length*devices[i].Cavity_Width*(1e-8)
            for j in range(len(devices[i].Current)):
                devices[i].Current[j] = devices[i].Current[j]/area
        current_label = "Current Density (mA/cm^2)"
    if plot == "IV":  
        for i in range(0,len(devices)):
            for j in range(0,len(temps)):
                if devices[i].Temperature == temps[j]:
                    if devices[i].Temperature != devices[i-1].Temperature or i==0:
                        plt.plot(devices[i].Voltage,devices[i].Current, color=colors[j], label= str(temps[j]) + ' C')
                    else:
                        plt.plot(devices[i].Voltage,devices[i].Current, color=colors[j])
        plt.xlabel('Voltage (V)')
        plt.title(devices[0].dev_name + ' on '+wafername)
        plt.ylabel(current_label)
        plt.grid()
        plt.legend()
        plt.show()
    elif plot == "LI":  
        for i in range(0,len(devices)):
            for j in range(0,len(temps)):
                if devices[i].Temperature == temps[j]:
                    if devices[i].Temperature != devices[i-1].Temperature or i==0:
                        plt.plot(devices[i].Current,devices[i].Power, color=colors[j], label= str(temps[j]) + ' C')
                    else:
                        plt.plot(devices[i].Current,devices[i].Power, color=colors[j])
        plt.xlabel(current_label)
        plt.title(devices[0].dev_name + ' on '+wafername)
        plt.ylabel('Power (mW)')
        plt.grid()
        plt.legend()
        plt.show()



def table_laser_data(devices):
    columns = ('Name','Temperature (C)' ,'Length (um)', 'Width (um)', 'Threshold Current (mA)', 'Slope')
    
    temps = []
    devices = sorted(devices, key=lambda x: x.Temperature)
    for i in range(0,len(devices)):
        temps= np.append(temps,devices[i].Temperature)
    rows = ['%d C' % x for x in temps]
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    cell_text= []
    for i in range(len(devices)):
        this_row = [devices[i].dev_name, devices[i].Temperature, devices[i].Cavity_Length,devices[i].Cavity_Width,devices[i].Threshold_Current,devices[i].Slope_Efficiency]
        cell_text.append(this_row)

    dataframe = pd.DataFrame(cell_text,columns=columns)
    display(dataframe)
    return(dataframe)



def plot_threshold_by_temp(dataframe, devices):
    df = dataframe
    for i in range(len(df)):
        area = df['Length (um)'][i]*df['Width (um)'][i]*(1e-8)
        df['Threshold Current (mA)'][i] = df['Threshold Current (mA)'][i]/area
    plt.semilogy(df['Temperature (C)'],df['Threshold Current (mA)'], 'o')
    T_0, intercept = np.polyfit(np.log(df['Threshold Current (mA)']),df['Temperature (C)'], 1)
    plt.xlabel('Temperature (C)')
    plt.title('Threshold v. Temperature')
    plt.ylabel('log(Threshold Current Density)')
    print('Characteristic Temperature: ' + str(T_0))


