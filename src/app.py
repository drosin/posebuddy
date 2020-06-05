import cv2
from timeit import default_timer as timer
import tkinter as tk
from tkinter import messagebox
import tk_tools
from src.prediction import Predictor
from src.prediction import Presence_Predictor
from src.handlers.config import load_yaml
from src.grab_images import get_images_from_webcam
from src.training import run_training


class App:
    def __init__(self, config_path='config/config.yaml'):
        self.config = load_yaml(config_path)
        self.update_period_ms = self.config['update_period_ms']
        self.predictor = Predictor(self.config['image_dir'], self.config['job_dir'])
        self.presence_predictor = Presence_Predictor(
            self.config['presence_image_dir'], self.config['presence_job_dir']
        )
        self.start_presence_time = timer()
        self.camera = cv2.VideoCapture(0)

        self.initialize_gui(
            app_name=self.config['app_name'], always_on_top=self.config['always_on_top']
        )
        self.gauge_yellow = 0.3
        self.gauge_red = 0.6
        self.initialize_gauge()
        self.initialize_led()
        self.initialize_train_button()

        self.get_prediction_event()
        self.gui.mainloop()

    def initialize_gui(self, app_name: str, always_on_top: bool):
        self.gui = tk.Tk()
        if always_on_top:
            self.gui.attributes('-topmost', True)
        self.gui.title(app_name)

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
        return
        self.led = tk_tools.Led(self.gui, size=50)
        self.led.pack()

    def initialize_train_button(self):
        self.button = tk.Button(self.gui, text='retrain', command=self.retrain)
        self.button.pack()

    def __del__(self):
        if 'camera' in locals():
            self.camera.release()

    def get_prediction_event(self):
        _, image = self.camera.read()
        bad_pose_prob = self.predictor.predict(image)
        presence_probability = self.presence_predictor.predict(image)
        print(
            f'\nBad posture prob: {str(round(bad_pose_prob,2))},\nPresence prob: {str(round(1-presence_probability,2))}'
        )
        self.update_gauge(bad_pose_prob)
        self.update_led(bad_pose_prob)
        self.alert_user_pause(presence_probability)
        self.gui.after(self.update_period_ms, self.get_prediction_event)

    def update_gauge(self, bad_pose_prob: float):
        self.gauge.set_value(
            # show 1% as lowest. Otherwise gauge gets weird.
            max(bad_pose_prob * 100, 1)
        )

    def update_led(self, bad_pose_prob: float):
        return
        if bad_pose_prob < self.gauge_yellow:
            self.led.to_green(on=True)
        elif bad_pose_prob < self.gauge_red:
            self.led.to_yellow(on=True)
        else:
            self.led.to_red(on=True)

    def alert_user_pause(self, presence_probability: float):
        if presence_probability > 0.5:
            self.start_presence_time = timer()
        work_time_sec = timer() - self.start_presence_time
        if work_time_sec > self.config['present_timer_sec']:
            print(
                "\n You have been sitting for too long, please stand up and walk around! \n"
            )
            # messagebox.showinfo(
            #     'Posture Alert!',
            #     'You have been sitting for too long, please stand up and walk around!',
            # )

    def retrain(self):
        pass


App()
