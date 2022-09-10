#!/bin/sh

#rm -rf build csfd.egg-info/ dist/

python3.7 setup.py bdist_wheel
python3.7 -m pip download --only-binary :all: . -d dist/wheels
kaggle datasets version --dir-mode zip -p dist/ -m ""
