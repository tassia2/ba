#!/bin/bash
# use bash as shell
#$ -S /bin/bash

BASE_DIR=$1
# use time to print the execution time, let the python create_node_folder script replace num_threads with the number of threads per node, wirte all output to the processlog.log
(/usr/bin/time -p mpirun -np num_threads $BASE_DIR/node_id/boussinesq2d $BASE_DIR/node_id/boussinesq2d.xml $BASE_DIR/node_id) &> $BASE_DIR/node_id/processlog.log

