import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


import json


import platform
import glob
import ntpath
import os

platform_system = platform.system()
file_separator=''
if(platform_system=='Windows'):
    file_separator = '\\'
else:#(platform_system=='Linux'):
    file_separator = '/'


n_time_points = 2976

path='.'+ file_separator
f_name = 'cv_3t.txt'
file= open(f_name,'r')

lines = json.load(file)
n_models = len(lines)/2
print("There are "+str(n_models)+" models to plot in this file.") #should be 4
print("Print the models all together")
colors = ['r','r','b','b','g','g']
fig1 = plt.figure(figsize=(8,8))
plt.xlabel('Epoch')
plt.ylabel('Loss (Cross entropy)')
for i in range(0,len(lines),2):
    model = lines[i][0]
    tr_loss = lines[i][2][2:]
    val_loss = lines[i+1][2][2:]
    epochs = range(2,len(tr_loss)+2)
    plt.plot(epochs,tr_loss,label=model+', train',color=colors[i],linestyle='--')
    plt.plot(epochs,val_loss,label=model+', validation',color=colors[i+1])
plt.legend()
plt.savefig(path+"loss_all_"+str(n_time_points)+".png")
fig2 = plt.figure(figsize=(8,8))
plt.xlabel('Epoch')
plt.ylabel('Classification error rate')
for i in range(0,len(lines),2):
    model = lines[i][0]
    tr_err = lines[i][3][2:]
    val_err = lines[i+1][3][2:]
    epochs = range(2,len(tr_err)+2)
    plt.plot(epochs,tr_err,label=model+', train',color=colors[i],linestyle='--')
    plt.plot(epochs,val_err,label=model+', validation',color=colors[i+1])
    plt.ylim((0,1.0))
plt.legend()
plt.savefig(path+"classErr_all_"+str(n_time_points)+".png")
