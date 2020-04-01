#!/usr/bin/bash

PROJECT_NAME="biharmonic"

[ "$1" == "-c" ] &&
[ -f "$PROJECT_NAME.h" ] && rm -rf "$PROJECT_NAME.h" && ffc -l dolfin $PROJECT_NAME.ufl &&
[ -d "build" ] && rm -rf build && mkdir build && cd build && cmake .. && make && cd ..

[ -d "results" ] && rm -rf results && mkdir results && cd results
../build/$PROJECT_NAME
[ ! -x "../view.py" ] && chmod +x view.py
../view.py "./${PROJECT_NAME}000000.vtu"