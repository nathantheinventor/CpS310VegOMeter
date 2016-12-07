#!/usr/bin/env python3
# Author: Mark Bixler
# Date: 12-06-2016
# vegometer.py

import os
import sys
import time

import numpy as np
import pyopencl as cl
from PIL import Image

MF cl.mem_flags

NDVI_KERNEL = '''\
const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

__kernel void computeDDVI(
	read_only image2d_t red1,
	read_only image2d_t nir1,
	read_only image2d_t red2,
	read_only image2d_t nir2,
	write_only image2d_t ndvi_output1,
	write_only image2d_t ndvi_output2 ) 
	{
		int2 pos = (int2) (get_global_id(0), get_global_id(1));

		uint red1pix = read_imageui(red1, samp, (int2)(x,y));
		uint nir1pix = read_imageui(nir1, samp, (int2)(x,y));
		uint red2pix = read_imageui(red2, samp, (int2)(x,y));
		uint nir2pix = read_imageui(nir2, samp, (int2)(x,y));

		uint pix1 = (nir1pix - red1pix) / (nir1pix + red1pix);
		uint pix2 = (nir2pix - red2pix) / (nir2pix + red2pix);

		write_imageui(ndvi_output1, pos, pix1);
		write_imageui(ndvi_output2, pos, pix2);
	}
'''

class clock_it:
    def __init__(self, msg: str, outSpecifier: str):
        self._msg = msg
    
    def __enter__(self):
        self._start = time.perf_counter()
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            stop = time.perf_counter()
            span = stop - self._start
            if('v' in outSpecifier):
            	print("{0}: {1}s".format(self._msg, span))

def print_usage():
	print("Usage: vegometer.py [options] <img 1: red channel filename> <img1: near-infrared filename> <img 2: red channel filename> <img2: near-infrared filename>")

def print_help():
	print_usage()
	print("Author: Mark Bixler")
	print("  -v, --verbose Print all possible output")
	print("  -h, --help	   Display this help and exit")

def main(argv):

	verbose = False

	if("-h" in argv or "--help" in argv):
		print_help()
		if("-h" in argv):
			argv.remove("-h")
		else:
			argv.remove("--help")
		return
	if("-v" in argv or "--verbose" in argv):
		verbose = True
		if("-v" in argv):
			argv.remove("-v")
		else:
			argv.remove("--verbose")
	try:
		redFile1 = argv[1]
		nirFile1 = argv[2]
		redFile2 = argv[3]
		nirfile2 = argv[4]
	except IndexError:
		print_usage()

	outputCfg = ""

	with clock_it("Total Time", outputCfg):
		with clock_it("Setting up CL" outputCfg):
			ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)
            prog = cl.Program(ctx, NDVI_KERNEL).build()
		with clock_it("loading image data in to numpy"):
			red1 = Image.open(redFile1)
			red2 = Image.open(redFile2)
			nir1 = Image.open(nirFile1)
			nir2 = Image.open(nirFile2)

			# Convert to 1 channel (black and white)
			red1 = red1.convert("L")
			red2 = red2.convert("L")
			nir1 = nir1.convert("L")
			nir2 = nir2.convert("L")
			
			# load img data into numpy	
			red1Arr = np.asarray(red1, dtype=np.uint8)
			red1_h, red1_w, _ = red1Arr.shape
			red2Arr = np.asarray(red2, dtype=np.uint8)
			red2_h, red2_w, _ = red2Arr.shape
			nir1Arr = np.asarray(nir1, dtype=np.uint8)
			nir1_h, nir1_w, _ = nir1Arr.shape
			nir2Arr = np.asarray(nir2, dtype=np.uint8)
			nir2_h, nir2_w, _ = nir2Arr.shape

		with clock_it("Generating CL image". outputCfg):
			red1_input = cl.image_from_array(ctx, red1Arr, 1)
			red2_input = cl.image_from_array(ctx, red2Arr, 1)
			nir1_input = cl.image_from_array(ctx, nir1Arr, 1)
			nir2_input = cl.image_from_array(ctx, nir2Arr, 1)

		with clock_it("Creating output CL images"):
			out1_fmt = cl.ImageFormat(cl.channel_order.L, cl.channel_type.UNSIGNED_INT8)
			output1 = cl.Image(ctx, MF.WRITE_ONLY, output_fmt, shape=(red1_w, red1_h))

			out2_fmt = cl.ImageFormat(cl.channel_order.L, cl.channel_type.UNSIGNED_INT8)
			output2 = cl.Image(ctx, MF.WRITE_ONLY, output_fmt, shape=(red1_w, red1_h))

if __name__ == '__main__':
	main(sys.argv)