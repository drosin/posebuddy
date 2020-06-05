import time
import os
import cv2
import tkinter as tk
from tkinter import messagebox
import tk_tools
from src.prediction import Predictor
from src.handlers.config import load_yaml
from src.grab_images import get_images_from_webcam
from src.training import run_training


class App:
    def __init__(self, config_path='config/config.yaml'):
        self.config = load_yaml(config_path)
        self.initialize_gui()
        self.gauge_yellow = 0.3
        self.gauge_red = 0.6
        self.initialize_gauge()
        self.initialize_led()
        self.initialize_train_button()

        self.camera = cv2.VideoCapture(0)
        if self.check_model_available():
            self.run_training()
        self.start_prediction_loop()
        self.gui.mainloop()

    def initialize_gui(self):
        self.gui = tk.Tk()
        if self.config['always_on_top']:
            self.gui.attributes('-topmost', True)
        self.gui.title(self.config['app_name'])

    def initialize_gauge(self):
        self.gauge = tk_tools.Gauge(
            self.gui,
            max_value=100.0,  # 100%
            label='probability',
            unit='%',
            yellow=self.gauge_yellow * 100,
            red=self.gauge_red * 100,
        )
        self.gauge.pack()

    def initialize_led(self):
        self.led = tk_tools.Led(self.gui, size=50)
        self.led.pack()

    def initialize_train_button(self):
        self.button = tk.Button(
            self.gui, text='retrain', command=self.retrain_button_pressed
        )
        self.button.pack()

    def __del__(self):
        if 'camera' in locals():
            self.camera.release()

    def start_prediction_loop(self):
        self.predictor = Predictor(self.config['image_dir'], self.config['job_dir'])
        self.loop_prediction = True
        self.start_prediction_event()

    def stop_prediction_loop(self):
        self.loop_prediction = False
        time.sleep(self.config['update_period_ms'] / 1000)
        # set form to stopped mode
        self.led.to_grey(on=False)
        self.led.to_grey(on=False)
        self.gauge.set_value(50)
        self.gui.update_idletasks()

    def start_prediction_event(self):
        _, image = self.camera.read()
        bad_pose_prob = self.predictor.predict(image)
        print(f'bad pose probability: {bad_pose_prob}')
        self.update_gauge(bad_pose_prob)
        self.update_led(bad_pose_prob)
        self.alert_user(bad_pose_prob)
        if self.loop_prediction:
            self.gui.after(self.config['update_period_ms'], self.start_prediction_event)

    def update_gauge(self, bad_pose_prob: float):
        self.gauge.set_value(
            # show 1% as lowest. Otherwise gauge gets weird.
            max(bad_pose_prob * 100, 1)
        )

    def update_led(self, bad_pose_prob: float):
        if bad_pose_prob < self.gauge_yellow:
            self.led.to_green(on=True)
        elif bad_pose_prob < self.gauge_red:
            self.led.to_yellow(on=True)
        else:
            self.led.to_red(on=True)

    def alert_user(self, bad_pose_prob: float):
        # messagebox.showinfo('Posture Alert!', 'Kindly sit straight! Thank you.')
        pass

    def check_model_available(self):
        model_dir_path = self.config['job_dir'] + '/models/'
        model_dir_exists = os.path.isdir(model_dir_path)
        return model_dir_exists and len(os.listdir(model_dir_path)) != 0

    def retrain_button_pressed(self):
        answer = messagebox.askquestion(
            'Retraining', 'Are you sure that you want to retrain'
        )
        if answer == 'yes':
            self.stop_prediction_loop()
            self.run_training()
            self.start_prediction_loop()

    def run_training(self):
        messagebox.showinfo(
            'Training', 'Please sit in a good position while moving around a bit.'
        )
        get_images_from_webcam(
            self.config['image_dir'],
            camera=self.camera,
            name='good',
            num_images=self.config['training_num_images'],
            sleep_time_s=self.config['training_update_period_s'],
        )
        messagebox.showinfo(
            'Training', 'Please sit in a bad position while moving around a bit.'
        )
        get_images_from_webcam(
            self.config['image_dir'],
            camera=self.camera,
            name='bad',
            num_images=self.config['training_num_images'],
            sleep_time_s=self.config['training_update_period_s'],
        )
        messagebox.showinfo('Training', 'Training starting. It will take a bit.')
        run_training(
            self.config['image_dir'],
            self.config['job_dir'],
            self.config['json_file_name'],
        )
        # messagebox.showinfo('Retraining', 'Retraining done.')


App()
