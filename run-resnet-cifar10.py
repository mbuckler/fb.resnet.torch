#! /usr/bin/env python

from subprocess import call
import sys 
import numpy as np
#from os import listdir
#from os.path import isfile, join

pipes_to_run = [0,1,2,3,4,5,6,7,8,9,10,11]
#pipes_to_run = [0]

#/datasets/cifar-10/
dataset_path = '/datasets/cifar-10/'

for vers in range(0,len(pipes_to_run)):

  print('Starting process for Version '+str(pipes_to_run[vers]))

  # Remove old training/testing data
  call("rm -rf gen/",shell=True)

  # Set path of data to compute on
  path = dataset_path+'v'+str(pipes_to_run[vers])+'/'

  # Compute mean and standard deviation for this version
  # Compute for subset of data for simplicity and speed
  print('Computing mean and standard deviation')
  
  filename   = path+'data_batch_2.bin'
  red_data   = np.empty([10000,32,32])
  green_data = np.empty([10000,32,32])
  blue_data  = np.empty([10000,32,32])

  with open(filename, 'rb') as f:
    for i in range(0,10000): # 10000
      # Read in the image label
      byte_s = f.read(1)
      for c in range(0,3):
        for y in range(0,32):
          for x in range(0,32):
            byte_s      = f.read(1)
            if c == 0: # Red
              red_data[i,x,y] = ord(byte_s[0])
            if c == 1: # Green
              green_data[i,x,y] = ord(byte_s[0])
            if c == 2: # Blue
              blue_data[i,x,y] = ord(byte_s[0])
  f.close()

  red_mean   = np.mean(red_data,   dtype=np.float32)
  red_std    = np.std (red_data,   dtype=np.float32)
  green_mean = np.mean(green_data, dtype=np.float32)
  green_std  = np.std (green_data, dtype=np.float32)
  blue_mean  = np.mean(blue_data,  dtype=np.float32)
  blue_std   = np.std (blue_data,  dtype=np.float32)
   
  # Write the result to file
  fout = open('mean_std.txt','w')
  fout.write(str(red_mean)+' '+str(green_mean)+' '+str(blue_mean)+' '+
             str(red_std)+' '+ str(green_std)+' '+ str(blue_std) ) 
  fout.close() 

  # Convert data to t7 files
  call("th /root/cifar.torch/Cifar10BinToTensor.lua -path /datasets/cifar-10/v"
         +str(pipes_to_run[vers])+"/", shell=True)

  print("Now running training for Version "+str(pipes_to_run[vers]))

  # Run the training and pipe output to a log file
  call("th main.lua -dataset cifar10 -data "+path+" -depth 20"
         +" 1> log_"+str(pipes_to_run[vers])+".txt", shell=True)

