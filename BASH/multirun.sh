#!/bin/bash

##Bash script to run a script multiple times

##run script , specifiy second number to determine how many runs
for i in {1..**number of iterations**};
do
./**filename** >> **output filename** &
wait
done 
wait 