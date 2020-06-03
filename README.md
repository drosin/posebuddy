# Posebuddy

This application will give you an alarm when you get into bad posture.

## How to use
### Setup your environment
- Install miniconda or anaconda.
- Create a new python 3.6 conda environment with `conda create -n posebuddy python=3.6`.
- Activate your conda environment with `conda activate posebuddy`.
- Install the dependencies with `pip install -f requirements.txt`.

### Run the code
- Run `python grab_images.py` to generate training data. While in good or bad posture, move a bit around.
- Run `python training.py` to train the classifier on your newly created training data.
- Run `python predict_image_stream.py` to see predictions.