from imageatm.components import DataPrep, Training, Evaluation
from src.handlers.image_json import create_training_json
from src.handlers.image_json import create_presence_training_json
from src.handlers.config import load_yaml


def run_training(image_dir, job_dir, json_file_name):
    """
    Runs the training of imageatm as described here:
    https://idealo.github.io/imageatm/examples/cats_and_dogs/
    """
    create_training_json(image_dir, json_file_name)
    dp = DataPrep(image_dir=image_dir, samples_file=json_file_name, job_dir=job_dir)
    dp.run(resize=True)
    trainer = Training(dp.image_dir, dp.job_dir, epochs_train_dense=5, epochs_train_all=5)
    trainer.run()
    e = Evaluation(image_dir=dp.image_dir, job_dir=dp.job_dir)
    e.run()


def run_presence_training(presence_image_dir, presence_job_dir, presence_json_file_name):
    """
    Runs the presence training of imageatm as described here:
    https://idealo.github.io/imageatm/examples/cats_and_dogs/
    """
    create_presence_training_json(presence_image_dir, presence_json_file_name)
    dp = DataPrep(
        image_dir=presence_image_dir,
        samples_file=presence_json_file_name,
        job_dir=presence_job_dir,
    )
    dp.run(resize=True)
    trainer = Training(dp.image_dir, dp.job_dir, epochs_train_dense=5, epochs_train_all=5)
    trainer.run()
    e = Evaluation(image_dir=dp.image_dir, job_dir=dp.job_dir)
    e.run()


if __name__ == '__main__':
    config = load_yaml('config/config.yaml')
    image_dir = config['image_dir']
    job_dir = config['job_dir']
    json_file_name = config['json_file_name']
    run_training(image_dir, job_dir, json_file_name)

    presence_image_dir = config['presence_image_dir']
    presence_job_dir = config['presence_job_dir']
    presence_json_file_name = config['presence_json_file_name']
    run_presence_training(presence_image_dir, presence_job_dir, presence_json_file_name)
