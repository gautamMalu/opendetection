#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"
tar -xvf "$DIR/mnist_png.tar.gz"

