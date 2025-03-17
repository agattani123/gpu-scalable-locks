## Scalable Lock Implementations: Evaluating GPU Concurrency Performance

# Check slides for Project motivation and pseudocode for different locks (spin, ticket, spinspin, anderson, mcs)
# We experiment with 5 different types of locks (code in the respective .h files)

## To compile and run different executables do the following:
# to compile, run:
# nvcc -o [filename] [filename].cu 
# replace [filename] with the filname of .cu file we are trying to compile

# To execute the .cu file, run:
# ./[filename]
# output includes each lock type, number of blocks counted (correctness), and time it took for program to run (performance)

# basicAtomicAddition.cu has the code for a simple atomic addition (no race-condition)
# lowComplexityCritical.cu has the locks ran on the small critical sections
# highComplecityCritical.cu has the locks ran on the high critical sections
