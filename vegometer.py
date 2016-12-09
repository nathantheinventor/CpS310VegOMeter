#!/usr/bin/env python3
# Author: Mark Bixler and Nathan Collins
# Date: 12-06-2016
# vegometer.py

import os
import sys
import time

import numpy as np
import pyopencl as cl
from PIL import Image

MF = cl.mem_flags

PYOPENCL_COMPILER_OUTPUT=1

NDVI_KERNEL = r'''\
const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

__kernel void computeNDVI(
    read_only image2d_t red1,
    read_only image2d_t nir1,
    read_only image2d_t red2,
    read_only image2d_t nir2,
    __global write_only float* ndvi_output1,
    read_only int width) {
        int2 pos = (int2) (get_global_id(0), get_global_id(1));
        
        int red1pix = read_imageui(red1, samp, pos).x;
        int nir1pix = read_imageui(nir1, samp, pos).x;
        int red2pix = read_imageui(red2, samp, pos).x;
        int nir2pix = read_imageui(nir2, samp, pos).x;
        
        float pix1 = (nir1pix - red1pix);
        float div1 = (nir1pix + red1pix) > 0 ? (float) (nir1pix + red1pix): 1.0;
        pix1 /= div1;
        
        float pix2 = (nir2pix - red2pix);
        float div2 = (nir2pix + red2pix) > 0 ? (float) (nir2pix + red2pix): 1.0;
        pix2 /= div2;
        
        float delta_pix = pix1 - pix2;
        
        ndvi_output1[pos.x * width + pos.y] = delta_pix;
    }
    
__kernel void threshold(
    __global read_only float* ndvi_input1,
    __global write_only int* output1,
    read_only float threshold,
    read_only int width) {
        int2 pos = (int2) (get_global_id(0), get_global_id(1));
        
        int coord = pos.x * width + pos.y;
        output1[coord] = ndvi_input1[coord] > threshold;
    }

__kernel void filter(
    __global read_only float* ndviData,
    __global read_only int* thresholdData,
    __global read_write float* output1,
    __global read_only int* convolutionData,
    read_only float multiplier,
    read_only int convWidth,
    read_only int width,
    read_only int height) {
        int2 pos = (int2) (get_global_id(0), get_global_id(1));
        
        int coord = pos.x * width + pos.y;
        
        int minus = (convWidth - 1) / 2;
        
        float sum = 0.0;
        int i = 0;
        for (int x = pos.x - minus; x <= pos.x + minus; x ++) {
            for (int y = pos.y - minus; y <= pos.y + minus; y ++) {
                int countThis = x >= 0 && x < width && y >= 0 && y < height;
                countThis = countThis && ((countThis) ? thresholdData[x * width + y]: 0);
                
                sum += (countThis ? ndviData[x * width + y] : 0.0) * convolutionData[i ++];
            }
        }
        
        sum *= multiplier;
        
        output1[coord] = sum;
    }

__kernel void collapse(
    __global read_only int* thresholds,
    __global write_only int* outputs,
    read_only int width) {
        int x = get_global_id(0);
        
        int count = 0;
        
        int pos = x * width;
        
        for (int i = 0; i < width; i ++) {
            count += thresholds[pos + i] ? 1: 0;
        }
        outputs[x] = count;
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
    print("Author: Mark Bixler and Nathan Collins")
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
            ctx = cl.create_some_context(interactive=False)
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
            
        with clock_it("Creating buffers", outputCfg):
            output1 = np.zeros((red1_h, red1_w), np.float32)
            output2 = np.zeros((red1_h, red1_w), np.int32)
            output3 = np.zeros((red1_h, red1_w), np.float32)
            output4 = np.zeros((red1_h,), np.int32)
            
            convolution = np.array(
                [[0, 1, 0],
                 [1, 4, 1],
                 [0, 1, 0]
                ], dtype = np.int32)
            
            convolutionBuff = cl.Buffer(ctx, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=convolution)
            
            ndviBuff        = cl.Buffer(ctx, MF.WRITE_ONLY, output1.nbytes)
            thresholdBuff   = cl.Buffer(ctx, MF.WRITE_ONLY, output2.nbytes)
            filterBuff      = cl.Buffer(ctx, MF.WRITE_ONLY, output3.nbytes)
            outBuff         = cl.Buffer(ctx, MF.WRITE_ONLY, output4.nbytes)
            
            
        with clock_it("Computing NDVI", outputCfg):
            with clock_it("running NDVI", outputCfg):
                prog.computeNDVI(queue, (red1_w, red1_h), None, red1_input, nir1_input, red2_input, nir2_input, ndviBuff, np.int32(red1_w)).wait()
            
            with clock_it("threshold 1", outputCfg):
                prog.threshold(queue, (red1_w, red1_h), None, ndviBuff, thresholdBuff, np.float32(0.5), np.int32(red1_w)).wait()
            
            with clock_it("filtering", outputCfg):
                prog.filter(queue, (red1_w, red1_h), None, ndviBuff, thresholdBuff, filterBuff, convolutionBuff, np.float32(0.125), np.int32(3), np.int32(red1_w), np.int32(red1_h)).wait()
            
            with clock_it("threshold 2", outputCfg):
                prog.threshold(queue, (red1_w, red1_h), None, filterBuff, thresholdBuff, np.float32(0.25), np.int32(red1_w)).wait()
            
            with clock_it("collapsing", outputCfg):
                prog.collapse(queue, (red1_h, ), None, thresholdBuff, outBuff, np.int32(red1_w)).wait()
            
            
            cl.enqueue_copy(queue, output4, outBuff)
            
            ans = 0
            for i in output4:
                ans += i
            print(ans)
            
            
if __name__ == '__main__':
    main(sys.argv)
