#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

pathsource=$1
pathtarget=$2
label=$3

passwordfile=~/.sshpass
user=tlampert
#host=elia.u-strasbg.fr
host=hpc-login.u-strasbg.fr
#host=fitz.u-strasbg.fr

#directories=[graphs,models]

mkdir -p ${pathtarget}
sshpass -f ${passwordfile} scp ${user}@${host}:${pathsource}/*.${label}.* ${pathtarget}
mkdir -p ${pathtarget}/graphs
sshpass -f ${passwordfile} scp ${user}@${host}:${pathsource}/graphs/*.${label}.* ${pathtarget}/graphs/
mkdir -p ${pathtarget}/models
sshpass -f ${passwordfile} scp ${user}@${host}:${pathsource}/models/*.${label}.* ${pathtarget}/models/
