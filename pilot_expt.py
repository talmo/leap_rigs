"""This script defines the pilot closed loop experiment.

Instructions:

1. Open cmder

2. Go to the right folder and activate environment
    cd C:\code\leap_rigs
    activate leap_rigs

Terminal should say this:
    C:\code\leap_rigs (main -> origin)
    (leap_rigs) λ

3. Run experiment with:
    python pilot_expt.py
"""

import leap_rigs
import datetime
import time
import numpy as np
import glob
import os
import h5py


##########
experiment_duration = 30  # minutes

daq_sample_frequency = 10000  # samples/s
cam_trigger_frequency = 150  # frames/s
callback_sample_frequency = 250  # samples (determines min opto latency)

cam_sn = "16276625" # MurthyLab-PC05 -> Cam1
ao_trigger = "Dev1/ao0"
ai_audio = "Dev1/ai0:8"
ai_exposure = "Dev1/ai15"
ao_opto = "Dev1/ao1"
ai_opto_loopback = "Dev1/ai9"

# cam_sn = "18159111" # MurthyLab-PC05 -> Cam2
# ao_trigger = "Dev1/ao2"
# ai_audio = "Dev1/ai16:24"
# ai_exposure = "Dev1/ai31"
# ao_opto = None
# ai_opto_loopback = None

model_paths = [
    "models/centroids.200823_193403.UNet.zip",  # fast
    # "centroid.200816_202746.UNet.zip",  # accurate
    "models/wt_gold.13pt.multiclass_topdown.zip",
]
##########


expt_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
data_path = f"D:/Motif/daq/daq.{expt_name}.h5"

# metadata = {"title": expt_name, "description": hostname}
metadata = {"title": expt_name}
metadata.update(leap_rigs.motif.get_experiment_metadata())

motif = leap_rigs.motif.get_motif_remote()


# Start camera
vid_filename = f"{expt_name}_{cam_sn}"
motif.call("camera/%s/recording/start" % cam_sn, codec="h264-gpu", filename=vid_filename, metadata=metadata)
print("Started recording camera video")

##########
stream_poller = leap_rigs.motif.StreamPoller(api=motif, camera_sn=cam_sn)
live_predictor = leap_rigs.tracking.LivePredictor.load_model(model_paths, get_image_fn=lambda: stream_poller.latest_image)

stream_poller.start()
live_predictor.start()

poses = leap_rigs.flies.PoseBuffer()

# opto_stim = leap_rigs.daq.test_opto_stim_fn
t_last_msg = time.perf_counter()
latencies = []
pose_preds = []
pose_samples = []
def opto_stim(s0, s1, number_of_samples, chunk_input_data, daq):
    global t_last_msg
    # img, md = stream_poller.latest_image
    # if img is not None:
    #     print(img.shape)
    # else:
    #     print(None)
    # return 0.0
    (img, meta), (pred, lp_timestamp) = live_predictor.last_data_and_prediction
    if meta is not None:
        # frame_idx, vr_timestamp = meta
        img_timestamp = meta["timestamp"]
        latency = lp_timestamp - img_timestamp
        latencies.append(latency)
        pose_preds.append(pred["instance_peaks"][0].copy())
        pose_samples.append(s0)

        # Update pose buffer and compute features.
        poses.update(pred)
        feats = poses.compute_features()

        # Decide trigger based on feature thresholds.
        # do_trigger = (feats.min_dist < 2) and (np.abs(feats.ang_f_rel_m) < 25)
        # do_trigger = (feats.min_dist < 2) and (np.abs(feats.ang_f_rel_m) < 25) and (np.abs(feats.ang_m_rel_f) > 120)
        do_trigger = (feats.min_dist < 2) and (np.abs(feats.ang_f_rel_m) < 25) and (np.abs(feats.ang_m_rel_f) > 145)

        msg = f"latency = {latency*1000:.1f} ms / min_dist = {feats.min_dist:.1f} mm / ang_f_rel_m = {feats.ang_f_rel_m:.1f} / ang_m_rel_f = {feats.ang_m_rel_f:.1f} / trigger: {do_trigger}"
        if (time.perf_counter() - t_last_msg) > 1.0:
            print(msg)
            t_last_msg = time.perf_counter()

        if do_trigger:
            return 3.0

        # Encode distance in opto output
        # if np.isnan(feats.dist):
        #     dist_norm = 0.
        # else:
        #     dist_norm = (np.clip(feats.dist, 0, 30) / 30 * 4) + 1
        # return dist_norm
    return 0.0
##########

# Setup DAQ
daq_controller = leap_rigs.daq.DAQController(
    ao_trigger=ao_trigger,
    ai_audio=ai_audio,
    ai_exposure=ai_exposure,
    ao_opto=ao_opto,
    ai_opto_loopback=ai_opto_loopback,
    data_path=data_path,
    # data_path=None,
    opto_data=opto_stim,
    # opto_data=0.,
    daq_sample_frequency=daq_sample_frequency,
    cam_trigger_frequency=cam_trigger_frequency,
    callback_sample_frequency=callback_sample_frequency,
    expected_duration=experiment_duration + 1,
)


# daq_controller.start()
daq_controller.setup_daq()
daq_controller.setup_saving()
print("Setup DAQ and saving")

# Start DAQ and triggering after a delay
daq_controller.start_saving()
print("Started saving")
time.sleep(2.5)
daq_controller.start_triggering()
print("Started triggering")


t0 = time.time()
done = False
while not done:
    # Check if we're past the max duration
    time_elapsed = time.time() - t0
    max_duration_expired = time_elapsed > (experiment_duration * 60)

    # Check if no cameras are running
    still_recording = motif.is_recording(cam_sn)
    
    # Determine if we're done
    done = max_duration_expired or (not still_recording)

    # Pause
    if not done:
        time.sleep(5.0)
        print(f"[t = {time_elapsed / 60:.2f} min] Still recording")
        # print(stream_poller.is_alive(), live_predictor.is_alive())

total_duration = time.time() - t0
print("Stopping experiment after %.1f minutes" % (total_duration / 60))

# Stop triggering
# daq_controller.stop()
daq_controller.stop_triggering()
print("Stopped triggering")

if len(latencies) > 0:
    print(f"Latencies: {np.mean(latencies)*1000:.1f} ms / Max: {max(latencies)*1000:.1f} ms / Min: {min(latencies)*1000:.1f} ms")

# Send stop signal to cameras
for cam in [cam_sn]:
    motif.call('camera/%s/recording/stop' % cam)
    print("STOP", cam)
    time.sleep(1)

# Wait for them to finish
done_recording = False
while not done_recording:
    # Check if cameras are running
    try:
        still_recording = [motif.is_recording(cam) for cam in [cam_sn]]
        done_recording = not any(still_recording)
    except:
        print("Error checking if cameras are still recording")
        done_recording = True


    if not done_recording:
        print("Waiting for cameras to finish recording...")
        time.sleep(1)

time.sleep(2.5)
daq_controller.stop_saving()
print("Stopped saving")

if daq_controller.is_saving:
    time.sleep(3)

    # Move data to final session folder
    with h5py.File(data_path, "r") as daqF:
        daq_data = daqF["data"]

        vidDest = "D:/Motif/" + vid_filename

        for _ in range(3):
            try:
                vidSource = glob.glob(f"D:/Motif/{cam_sn}/{vid_filename}*")[0]
                os.rename(vidSource, vidDest)
                print(f"Moved: {vidSource} -> {vidDest}")
                break
            except:
                time.sleep(3)

        with h5py.File(vidDest + "/daq.h5", "w") as f:
            f.create_dataset("audio",data=daq_data[0:9,:], compression="gzip", compression_opts=1)
            f.create_dataset("sync",data=daq_data[9,:], compression="gzip", compression_opts=1)
            print("Saved audio and sync")
            if daq_data.shape[0] > 10:
                f.create_dataset("opto", data=daq_data[10,:], compression="gzip", compression_opts=1)
                print("Saved opto outputs")
            if len(latencies) > 0:
                f.create_dataset("pose_latencies", data=np.array(latencies), compression="gzip", compression_opts=1)
                print("Saved pose latencies")
            if len(pose_preds) > 0:
                f.create_dataset("pose_preds", data=np.array(pose_preds), compression="gzip", compression_opts=1)
                print("Saved pose predictions")
            if len(latencies) > 0:
                f.create_dataset("pose_samples", data=np.array(pose_samples), compression="gzip", compression_opts=1)
                print("Saved pose sample inds")

        print("Moved data to final session folder:", vidDest)
