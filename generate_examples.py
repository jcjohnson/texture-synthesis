import os, subprocess

SIZE = 512
KERNEL_SIZES = [3, 5, 7, 9, 11, 13, 15, 17]
INPUT_DIR = 'examples/inputs'
OUTPUT_DIR = 'examples/outputs'

for filename in os.listdir(INPUT_DIR):
  basename, ext = os.path.splitext(filename)
  input_file = os.path.join(INPUT_DIR, filename)
  for k in KERNEL_SIZES:
    output_file = '%s_%d_k%d%s' % (basename, SIZE, k, ext)
    output_file = os.path.join(OUTPUT_DIR, output_file)
    cmd = ('th synthesis.lua -source %s -output_file %s '
           '-height %d -width %d -k %d') % (
               input_file,
               output_file,
               SIZE,
               SIZE,
               k)
    print cmd
    subprocess.call(cmd, shell=True)
