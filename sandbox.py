"""This module contains code for testing individual submodules."""

from time import perf_counter, sleep
import cv2
import numpy as np
import threading
from typing import Tuple, Optional, Union, List, Dict, Callable, Any

from leap_rigs.tracking import VideoReader, LivePredictor
from leap_rigs.flies import compute_features, COL_F, COL_M

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def test_video_reader():
    vr = VideoReader("tests/data/test.mp4", fps=200)
    print(vr)

    vr.start()
    while vr.is_alive():
        img, (frame_idx, timestamp) = vr.last_data
        sleep(0.2)
        effective_fps = (frame_idx+1) / timestamp if frame_idx is not None else None
        print(f"frame_idx = {frame_idx} / ts = {timestamp} = {effective_fps}")
    print(vr)


def test_predictor():
    vr = VideoReader("tests/data/test.mp4", fps=150)

    model_paths = [
        "models/centroids.200823_193403.UNet.zip",
        "models/wt_gold.13pt.multiclass_topdown.zip",
    ]
    lp = LivePredictor.load_model(model_paths, get_image_fn=lambda: vr.last_data)

    pred = lp.predict(np.zeros(vr.shape[1:], dtype="uint8"))
    # print(pred)
    # print("Initialized:", lp.predictor.inference_model.input_shapes)
    # print(lp.predictor.inference_model.input_shape)

    # pred = lp.predict(vr.image)
    # print(pred)

    vr.start()
    lp.start()

    vr_timestamps = []
    lp_timestamps = []
    latencies = []
    while lp.is_alive():
        # img, frame_idx, timestamp = vr.last_data
        (img, meta), (pred, lp_timestamp) = lp.last_data_and_prediction
        sleep(0.1)
        if meta is not None:
            frame_idx, vr_timestamp = meta
            latency = lp_timestamp - vr_timestamp

            vr_timestamps.append(vr_timestamp)
            lp_timestamps.append(lp_timestamp)
            latencies.append(latency)
            # print(frame_idx, vr_timestamp, lp_timestamp)
            print(frame_idx, latency, 1/latency)
    latencies = np.array(latencies)
    print(latencies.mean(), latencies.min(), latencies.max(), 1 / latencies.mean())


def setup_viz_plot(img_shape):
    fig = plt.figure(figsize=(6, 6.5))
    ax = plt.axes([0, 0, 1, 0.95])
    h_img = plt.imshow(np.zeros(img_shape, dtype="uint8").squeeze(), cmap="gray", vmin=0, vmax=255)
    h_pts1, = plt.plot(np.nan, np.nan, ".", c=COL_F)
    h_pts2, = plt.plot(np.nan, np.nan, ".", c=COL_M)
    h_title = plt.title("Initializing...")
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    def close(event):
        if event.key == "q":
            plt.close(event.canvas.figure)
    cid = fig.canvas.mpl_connect("key_press_event", close)

    return fig, h_img, h_title, h_pts1, h_pts2


def test_live_viz():
    vr = VideoReader("tests/data/test.mp4", fps=50)

    fig, h_img, h_title, h_pts1, h_pts2 = setup_viz_plot(vr.shape[1:])

    def update(i):
        img, frame_idx, timestamp = vr.last_data
        if img is not None:
            print(frame_idx)
            h_img.set_data(img.squeeze())
            h_title.set_text(f"frame_idx = {frame_idx} / timestamp = {timestamp}")
        # else:
            # plt.close(fig)

    ani = FuncAnimation(fig, update, interval=200)
    vr.start()
    plt.show()


def test_live_inference_viz():
    vr = VideoReader("tests/data/test.mp4", fps=30)
    model_paths = [
        "models/centroids.200823_193403.UNet.zip",  # fast
        # "models/centroid.200816_202746.UNet.zip",  # accurate
        "models/wt_gold.13pt.multiclass_topdown.zip",
    ]
    lp = LivePredictor.load_model(model_paths, get_image_fn=lambda: vr.last_data)
    pred = lp.predict(np.zeros(vr.shape[1:], dtype="uint8"))


    fig, h_img, h_title, h_pts1, h_pts2 = setup_viz_plot(vr.shape[1:])

    def update(i):
        (img, meta), (pred, lp_timestamp) = lp.last_data_and_prediction
        if meta is not None:
            frame_idx, vr_timestamp = meta
            latency = lp_timestamp - vr_timestamp
            print(frame_idx, latency)

            h_img.set_data(img.squeeze())
            pts1, pts2 = pred["instance_peaks"][0]
            h_pts1.set_data(pts1[:, 0], pts1[:, 1])
            h_pts2.set_data(pts2[:, 0], pts2[:, 1])
            h_title.set_text(f"frame_idx = {frame_idx} / timestamp = {vr_timestamp:.1f} s / latency = {latency*1000:.1f} ms")
        # else:
            # plt.close(fig)

    vr.start()
    lp.start()
    ani = FuncAnimation(fig, update, interval=100)
    plt.show()


def test_cl_viz():
    vr = VideoReader("tests/data/test.mp4", fps=30)

    model_paths = [
        "models/centroids.200823_193403.UNet.zip",  # fast
        # "models/centroid.200816_202746.UNet.zip",  # accurate
        "models/wt_gold.13pt.multiclass_topdown.zip",
    ]
    lp = LivePredictor.load_model(model_paths, get_image_fn=lambda: vr.last_data)
    pred = lp.predict(np.zeros(vr.shape[1:], dtype="uint8"))


    fig, h_img, h_title, h_pts1, h_pts2 = setup_viz_plot(vr.shape[1:])
    

    def update(i):
        (img, meta), (pred, lp_timestamp) = lp.last_data_and_prediction
        if meta is not None:
            frame_idx, vr_timestamp = meta
            latency = lp_timestamp - vr_timestamp

            pts1, pts2 = pred["instance_peaks"][0]
            feats = compute_features(pts1, pts2)
            
            side_f = "N/A"
            if ~np.isnan(feats.ang_f_rel_m):
                side_f = "LEFT" if feats.ang_f_rel_m < 0 else "RIGHT"

            msg = f"dist = {feats.dist:.1f} / ang = {feats.ang_f_rel_m:.1f} ({side_f})"
            print(msg)

            h_img.set_data(img.squeeze())
            h_pts1.set_data(pts1[:, 0], pts1[:, 1])
            h_pts2.set_data(pts2[:, 0], pts2[:, 1])
            h_title.set_text(msg)
        # else:
            # plt.close(fig)

    vr.start()
    lp.start()

    ani = FuncAnimation(fig, update, interval=1000)
    plt.show()


def test_cl_trigger_viz():
    vr = VideoReader("tests/data/test.mp4", fps=30)

    model_paths = [
        "models/centroids.200823_193403.UNet.zip",  # fast
        # "models/centroid.200816_202746.UNet.zip",  # accurate
        "models/wt_gold.13pt.multiclass_topdown.zip",
    ]
    lp = LivePredictor.load_model(model_paths, get_image_fn=lambda: vr.last_data)
    pred = lp.predict(np.zeros(vr.shape[1:], dtype="uint8"))


    fig, h_img, h_title, h_pts1, h_pts2 = setup_viz_plot(vr.shape[1:])

    class PoseBuffer:
        def __init__(self):
            self.last_pose_m = None
            self.last_pose_f = None
            self.pose_m = None
            self.pose_f = None

        def update(self, pred):
            self.pose_f, self.pose_m = pred["instance_peaks"][0]
            if self.last_pose_f is None or (~np.isnan(self.pose_f)).any():
                self.last_pose_f = self.pose_f
            if self.last_pose_m is None or (~np.isnan(self.pose_m)).any():
                self.last_pose_m = self.pose_m

        def compute_features(self):
            return compute_features(self.last_pose_f, self.last_pose_m)
    
    poses = PoseBuffer()

    def update(i):
        (img, meta), (pred, lp_timestamp) = lp.last_data_and_prediction
        if meta is not None:
            frame_idx, vr_timestamp = meta
            latency = lp_timestamp - vr_timestamp

            poses.update(pred)
            feats = poses.compute_features()

            do_trigger = (feats.min_dist < 2) and (np.abs(feats.ang_f_rel_m) < 25)

            msg = f"min_dist = {feats.min_dist:.1f} mm / ang = {feats.ang_f_rel_m:.1f} / trigger: {do_trigger}"
            print(msg)

            h_img.set_data(img.squeeze())
            h_pts1.set_data(poses.pose_f[:, 0], poses.pose_f[:, 1])
            h_pts2.set_data(poses.pose_m[:, 0], poses.pose_m[:, 1])
            h_title.set_text(msg)
        # else:
            # plt.close(fig)

    vr.start()
    lp.start()

    ani = FuncAnimation(fig, update, interval=100)
    plt.show()


if __name__ == "__main__":
    # test_video_reader()
    # test_predictor()
    # test_live_viz()
    # test_live_inference_viz()
    # test_cl_viz()
    test_cl_trigger_viz()
