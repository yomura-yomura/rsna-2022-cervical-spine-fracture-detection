#!/bin/sh

python3.7 setup.py bdist_wheel
python3.7 -m pip download --only-binary :all: -r requirements.txt -d dist/wheels
kaggle datasets version --dir-mode zip -p dist/ -m ""
