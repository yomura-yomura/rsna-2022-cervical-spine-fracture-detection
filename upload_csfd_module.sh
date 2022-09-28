#!/bin/sh

python3.7 setup.py bdist_wheel
python3.7 -m pip download --only-binary :all: . -d dist/wheels
kaggle datasets version --dir-mode skip -p dist/ -m ""
#kaggle datasets version -p dist/wheels -m ""
