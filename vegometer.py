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

MF = cl.mem_flags

NDVI_KERNEL = '''\
const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

__kernel void computeNDVI(
	read_only image2d_t red1,
	read_only image2d_t nir1,
	write_only image2d_t ndvi_output1 ) 
	{
		int2 pos = (int2) (get_global_id(0), get_global_id(1));

		uint4 red1pix = read_imageui(red1, samp, (int2)(pos.x,pos.y));
		uint4 nir1pix = read_imageui(nir1, samp, (int2)(pos.x,pos.y));

		uint4 pix1 = (0,0,0,0);
		pix1 = (nir1pix - red1pix) / (nir1pix + red1pix);

		write_imageui(ndvi_output1, pos, pix1);
	}
'''

class clock_it:
	def __init__(self, msg: str, outSpecifier: str):
		self._msg = msg
		self.outSpec = outSpecifier
	
	def __enter__(self):
		self._start = time.perf_counter()
	
	def __exit__(self, exc_type, exc_value, traceback):
		if exc_type is None:
			stop = time.perf_counter()
			span = stop - self._start
			if('v' in self.outSpec):
				print("{0}: {1}s".format(self._msg, span))

def print_usage():
	print("Usage: vegometer.py [options] <img 1: red channel filename> <img1: near-infrared filename> <img 2: red channel filename> <img2: near-infrared filename>")

def print_help():
	print_usage()
	print("Author: Mark Bixler")
	print("  -v, --verbose Print all possible output")
	print("  -h, --help	   Display this help and exit")

def main(argv):

	outputCfg = ""

	if("-h" in argv or "--help" in argv):
		print_help()
		if("-h" in argv):
			argv.remove("-h")
		else:
			argv.remove("--help")
		return
	if("-v" in argv or "--verbose" in argv):
		outputCfg += 'v'
		if("-v" in argv):
			argv.remove("-v")
		else:
			argv.remove("--verbose")
	try:
		redFile1 = argv[1]
		nirFile1 = argv[2]
		redFile2 = argv[3]
		nirFile2 = argv[4]
	except IndexError:
		print_usage()
		return

	with clock_it("Total Time", outputCfg):
		with clock_it("Setting up CL", outputCfg):
					ctx = cl.create_some_context()
					queue = cl.CommandQueue(ctx)
					prog = cl.Program(ctx, NDVI_KERNEL).build()
		with clock_it("loading image data in to numpy", outputCfg):
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
			red1_h, red1_w = red1Arr.shape
			red2Arr = np.asarray(red2, dtype=np.uint8)
			red2_h, red2_w = red2Arr.shape
			nir1Arr = np.asarray(nir1, dtype=np.uint8)
			nir1_h, nir1_w = nir1Arr.shape
			nir2Arr = np.asarray(nir2, dtype=np.uint8)
			nir2_h, nir2_w = nir2Arr.shape

		with clock_it("Generating CL image", outputCfg):
			red1_input = cl.image_from_array(ctx, red1Arr, 1)
			red2_input = cl.image_from_array(ctx, red2Arr, 1)
			nir1_input = cl.image_from_array(ctx, nir1Arr, 1)
			nir2_input = cl.image_from_array(ctx, nir2Arr, 1)

		with clock_it("Creating output CL images", outputCfg):
			out1_fmt = cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.UNSIGNED_INT8)
			output1 = cl.Image(ctx, MF.WRITE_ONLY, out1_fmt, shape=(red1_w, red1_h))

			out2_fmt = cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.UNSIGNED_INT8)
			output2 = cl.Image(ctx, MF.WRITE_ONLY, out1_fmt, shape=(red2_w, red2_h))

		with clock_it("Computing NDVI", outputCfg):
			with clock_it("running NDVI kernel on first image", outputCfg):
				prog.computeNDVI(queue, (red1_w, red1_h), None, red1_input, nir1_input, output1)

			out1_array = np.empty_like(red1Arr)
			cl.enqueue_copy(queue, out1_array, output1, origin=(0,0), region=(red1_w,red1_h))

			ndvi1_img = Image.fromarray(out1_array).convert("L")

			with clock_it("Running kerenl on second image", outputCfg):
				prog.computeNDVI(queue, (red2_w, red2_h), None, red2_input, nir2_input, output2)

			out2_array = np.empty_like(red2Arr)
			cl.enqueue_copy(queue, out2_array, output2, origin=(0,0), region=(red2_w,red2_h))

			ndvi2_img = Image.fromarray(out2_array).convert("L")

if __name__ == '__main__':
	main(sys.argv)
