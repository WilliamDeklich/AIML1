#!/usr/bin/python3

#importing jetson modules for recognizing and loading images
import jetson_inference
import jetson_utils
#parsing command line
import argparse

#parses the images file name andselct a network (next 4 lines)
#defines expected command line args and parses them when script runs
parser = argparse.ArgumentParser()
#the filename that we pass into the command line 
parser.add_argument("filename", type = str, help = "filename of image to process")
#adds network
parser.add_argument("--network", type = str, default = "googlenet", help = "model to use where google net is the deafult")
#takes in args from commmand line and converts into object "opt"
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)

#load the recognition network from command line
net = jetson_inference.imageNet(opt.network)

class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)

print("image is recognized as " + str(class_desc) +"(class # " + str(class_idx) + ") with " +str(confidence *100)+ "% confidence")