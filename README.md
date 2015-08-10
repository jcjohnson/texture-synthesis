# texture-synthesis

This is a Torch implementation of a texture-synthesis algorithm very similar to [1].

Given an input texture patch, the algorithm generates a larger version of the same texture. Here's an example:

### Input
<img src="https://github.com/jcjohnson/texture-synthesis/blob/master/examples/inputs/scales.png?raw=true">

### Output
<img src="https://github.com/jcjohnson/texture-synthesis/blob/master/examples/outputs/scales_512_k13.png?raw=true">

## Usage
Texture synthesis is implemented in the script `synthesis.lua`. The following command line options are available:
* `-source`: Path to the source image.
* `-output_file`: Path where the output should be written.
* `-height`: Height of the output file, in pixels.
* `-width`: Width of the output file, in pixels.
* `-k`: Kernel size; must be an odd integer.
* `-gpu`: Which GPU to use. Setting `gpu >= 0` will run in GPU mode, and setting `gpu < 0` will run in CPU-only mode.

# Works Cited:
[1] Efros, Alexei, and Thomas K. Leung. "Texture synthesis by non-parametric sampling." ICCV 1999.
