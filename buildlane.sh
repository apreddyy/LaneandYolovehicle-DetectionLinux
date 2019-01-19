#!/bin/bash

rm -r build
mkdir build
cp calibration.yml build
cp graph-vehicledetection.pb build
cp openv.avi build 
cd build
cmake ..
make
./lanedetection

