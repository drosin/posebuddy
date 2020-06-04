import os
import json


def create_training_json(image_dir, json_file_name):
    """
    create json for training as described here: 
    https://idealo.github.io/imageatm/examples/cats_and_dogs/
    """
    filenames = os.listdir(image_dir)
    sample_json = []
    for filename in filenames:
        sample_json.append(
            {'image_id': filename, 'label': 'good' if 'good' in filename else 'bad'}
        )

    with open(json_file_name, 'w') as outfile:
        json.dump(sample_json, outfile, indent=4, sort_keys=True)
