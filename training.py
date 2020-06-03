from imageatm.components import DataPrep, Training, Evaluation
import os
import json

job_dir = 'train_job'
image_dir = 'train_job/images'
json_file_name = 'data.json'

# create json for training
filenames = os.listdir(image_dir)
sample_json = []
for filename in filenames:
    sample_json.append(
        {'image_id': filename, 'label': 'good' if 'good' in filename else 'bad'}
    )

with open(json_file_name, 'w') as outfile:
    json.dump(sample_json, outfile, indent=4, sort_keys=True)


dp = DataPrep(image_dir=image_dir, samples_file=json_file_name, job_dir=job_dir)

dp.run(resize=True)

trainer = Training(dp.image_dir, dp.job_dir, epochs_train_dense=5, epochs_train_all=5)

trainer.run()

e = Evaluation(image_dir=dp.image_dir, job_dir=dp.job_dir)

e.run()
