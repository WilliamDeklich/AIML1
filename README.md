# My-segnet-final
The project uses segmentation to discern different objects and outputs a mask using different colors to visualise where the objects are on screen. This project in particular is able to detect cars, buses, people, lights, signs, the road, the sky, and trees.

![add image descrition here](direct image link here)

## The Algorithm

#!/usr/bin/env python3

import sys
import argparse

from jetson_inference import segNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log

from segnet_utils import *

# parse the command line
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=segNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

#this block of code alllows fokr more customization while running the program
parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="WHich network to use")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=150.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
parser.add_argument("--stats", action="store_true", help="compute statistics about segmentation mask class output")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# load the segmentation network
net = segNet(args.network, sys.argv)


# sets the opacity of the mask
net.SetOverlayAlpha(args.alpha)

# makes the photo/video of the output
output = videoOutput(args.output, argv=sys.argv)

# create buffer manager
buffers = segmentationBuffers(net, args)

# create video source
input = videoSource(args.input, argv=sys.argv)

# analyzing the photo/video
while True:
    # capture the next image
    img_input = input.Capture()

    if img_input is None: # timeout
        continue
        
    # allocate buffers for this size image
    buffers.Alloc(img_input.shape, img_input.format)

    # process the segmentation network
    net.Process(img_input, ignore_class=args.ignore_class)

    # generate the overlay
    if buffers.overlay:
        net.Overlay(buffers.overlay, filter_mode=args.filter_mode)

    # generate the mask
    if buffers.mask:
        net.Mask(buffers.mask, filter_mode=args.filter_mode)

    # composite the images
    if buffers.composite:
        cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
        cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)

    # render the output image
    output.Render(buffers.output)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    cudaDeviceSynchronize()
    net.PrintProfilerTimes()

    # compute segmentation class stats
    if args.stats:
        buffers.ComputeStats()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
## Running this project
after opening the project do the followuing in the terminal

cd jetson-inference
docker/run.sh
enter the password
cd build
cd aarch64
cd bin
./segnet.py --network=fcn-resnet18-cityscapes images/city_0.jpg images/test/output.jpg

This should open the file "city_0.jpg" in the directory build/aarch64/bin/images/test, and create a new image called "output.jpg" in the same directory. If the code runs but no new image shows up, delete the inital photo, which should cause the new image to appear



[View a video explanation here](video link)
