from imageatm.components import DataPrep, Training, Evaluation
from src.handlers.image_json import create_training_json


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


if __name__ == '__main__':
    image_dir = 'train_job/images'
    job_dir = 'train_job'
    json_file_name = 'data.json'
    run_training(image_dir, job_dir, json_file_name)
