#!/usr/bin/bash
# to compile and run: ./run.sh -c
# to just run       : ./run.sh

PROJECT_NAME="navier-stokes"

## if -c is provided change the project in cmakelists to $PROJECT_NAME
[ "$1" == "-c" ] &&
sed -i "s/\(set[(]PROJECT_NAME \)\(.*\)\([)]\)/\1 $PROJECT_NAME\3/g" CMakeLists.txt

## if -c is provided compile the ufl file
[ "$1" == "-c" ] &&
ffc -l dolfin *.ufl

## if -c is provided delete the old build folder
[ "$1" == "-c" ] &&
[ -d "build" ] && rm -rf build 

## if -c is provided make build directory, and compile the code
[ "$1" == "-c" ] &&
mkdir build && cd build && echo "localhost slots=10" > hostfile && cmake .. && make && cd ..

## delete results folder if exists
[ -d "results" ] && rm -rf results

## move results to a different folder
# [ -d "results" ] && mv results results.old

## make results folder and run the executable
mkdir results && cd results

export OMP_NUM_THREADS=1
mpirun --hostfile ../build/hostfile -n 5 ../build/$PROJECT_NAME
