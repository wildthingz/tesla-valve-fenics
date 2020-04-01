#!/usr/bin/bash
[ -f "biharmonic.h" ] && rm -rf "biharmonic.h"
ffc -l dolfin biharmonic.ufl
[ -d "build" ] && rm -rf build
mkdir build && cd build && cmake .. && make && cd ..
[ -d "results" ] && mv results results.old
mkdir results && cd results
../build/biharmonic