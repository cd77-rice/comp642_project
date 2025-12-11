# COMP642 - Project

**Machine learning**

Student: Christof Dittmar

ID: cd77

## Introduction

This is the final project for the course COMP642.

Task: Binary classification of images into

1. Human on image
2. No human on image

The resulting model is meant to be used for a trail camera setup in order
to discard images of humans and only keep images of wildlife.

## How to run

- Build the conda env with `conda env create -f env.yml`
- Activate the environment by `conda activate trail_detector`
- Generate images by `python generate.py`
- Run the training and testing by `python run.py`
