#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e

EXAMPLE=examples/mnist
DATA=examples/mnist/mnist_png/
BUILD=build/examples/mnist

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf $EXAMPLE/mnist_png/train_${BACKEND}
rm -rf $EXAMPLE/mnist_png/test_${BACKEND}

$BUILD/convert_mnist_data $DATA 

echo "Done."
