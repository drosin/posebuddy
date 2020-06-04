# Posebuddy

This application will give you an alarm when you get into bad posture.

## How to use
### Setup your environment
- Install miniconda or anaconda.
- Create a new python 3.6 conda environment with `conda create -n posebuddy python=3.6`.
- Activate your conda environment with `conda activate posebuddy`.
- Install the dependencies with `pip install -f requirements.txt`.

### Run capture, train, and predict individually
- Run `python -m src.grab_images` to generate training data. While in good or bad posture, move a bit around.
- Run `python -m src.training` to train the classifier on your newly created training data.
- Run `python -m src.predict_image_stream` to see fetch pictures from your webcam to evaluate your pose continuously.

### Run the app after the above steps for training have been done
- Run `python -m src.app`