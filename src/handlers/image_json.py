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


def create_presence_training_json(presence_image_dir, presence_json_file_name):
    """
    create json for user presence training as described here: 
    https://idealo.github.io/imageatm/examples/cats_and_dogs/
    """
    presence_filenames = os.listdir(presence_image_dir)
    sample_json = []
    for filename in presence_filenames:
        sample_json.append(
            {'image_id': filename, 'label': 'present' if 'present' in filename else 'away'}
        )

    with open(presence_json_file_name, 'w') as outfile:
        json.dump(sample_json, outfile, indent=4, sort_keys=True)
