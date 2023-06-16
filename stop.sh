#!/bin/bash
for process in $(seq 21945 21954)
do
    echo $process
    kill $process
done
