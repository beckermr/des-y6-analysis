#!/usr/bin/env bash
pushd jobs
for fname in `ls *.sub`; do
  wq sub -b $fname
done
popd
