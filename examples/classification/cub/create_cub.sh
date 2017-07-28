#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e

EXAMPLE=examples/classification/cub
BUILD=build/examples/classification/cub

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf $EXAMPLE/data/train_${BACKEND}
rm -rf $EXAMPLE/data/test_${BACKEND}

$BUILD/convert_cub

echo "Done."
