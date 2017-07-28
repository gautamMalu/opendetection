#!/usr/bin/env sh
# This scripts downloads the CUB data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

mkdir data
# Download the files
wget -P ./data/ http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz
wget -P ./data/ http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz

#unzips the files
tar -xvf ./data/images.tgz -C ./data
tar -xvf ./data/lists.tgz -C ./data

# creating train.txt and test.txt from the default files
awk -F. '{print $0,$1}' ./data/lists/train.txt > ./data/train.txt
awk -F. '{print $0,$1}' ./data/lists/test.txt > ./data/test.txt

#get the caffenet model
wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
