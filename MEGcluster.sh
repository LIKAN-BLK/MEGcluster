#!/bin/sh
#SBATCH -D /s/ls2/u-sw/likan_blk/MEGcluster
#SBATCH -o out/%j.out
#SBATCH -e out/%j.err
#SBATCH -t 02:30:00
#SBATCH -p hpc2-16g-3d
#SBATCH --cpus-per-task 8

/s/ls2/u-sw/likan_blk/anaconda2/bin/python2.7 main.py $1
