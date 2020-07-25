#!/usr/bin/bash
# to compile and run: ./run.sh -c
# to compile only: ./run.sh -co
# to just run       : ./run.sh

PROJECT_NAME="test6"

## if -c is provided change the project in cmakelists to $PROJECT_NAME
[ "$1" == "-c" ] || [ "$1" == "-co" ] &&
sed -i "s/\(set[(]PROJECT_NAME \)\(.*\)\([)]\)/\1 $PROJECT_NAME\3/g" CMakeLists.txt

## if -c is provided compile the ufl file
count=`ls -1 *.ufl 2>/dev/null | wc -l`
[ "$1" == "-c" ] || [ "$1" == "-co" ] && [ $count != 0 ] && 
ffc -l dolfin *.ufl


## if -c is provided delete the old build folder
[ "$1" == "-c" ] || [ "$1" == "-co" ] &&
[ -d "build" ] && rm -rf build 

## if -c is provided make build directory, and compile the code
[ "$1" == "-c" ] || [ "$1" == "-co" ] &&
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=release .. && make -j 4 && cd ..

## delete results folder if exists
[ -d "results" ] && rm -rf results

## move results to a different folder
# [ -d "results" ] && mv results results.old

## make results folder and run the executable
mkdir results && cd results

[ "$1" != "-co" ] &&
echo "localhost slots=10" > hostfile &&
export OMP_NUM_THREADS=1 &&
mpirun --hostfile hostfile -n 4 ../build/$PROJECT_NAME

