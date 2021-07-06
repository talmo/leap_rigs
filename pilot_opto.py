"""This script defines the pilot opto experiment.

Instructions:

1. Open cmder

2. Go to the right folder and activate environment
    cd C:\code\leap_rigs
    activate leap_rigs

Terminal should say this:
    C:\code\leap_rigs (main -> origin)
    (leap_rigs) λ

3. Run experiment with:
    python pilot_opto.py
"""


###################################################
############## Experiment parameters ##############
###################################################
experiment_duration = 20/60  # minutes

daq_sample_frequency = 10000  # samples/s
cam_trigger_frequency = 150  # frames/s
callback_sample_frequency = 250  # samples (determines min opto latency -- not important in open loop)

# Set these to True or False to use one or two cameras
record_left_camera = True
record_right_camera = True

# Set these to None to disable opto
# opto_stim_left = None
# opto_stim_right = None

# Or set them to a path to a MAT file with pre-generated stimulus.
#
# These are created in MATLAB like:
# >> stim = [ones(10000, 1) * 3; zeros(10000, 1)]; save('opto_stims/example_opto_stim1.mat', 'stim')
# >> stim = [zeros(10000, 1); ones(10000, 1) * 3]; save('opto_stims/example_opto_stim2.mat', 'stim')
#
# You can also use the same one for both cameras.
opto_stim_left = "opto_stims/example_opto_stim1.mat"
opto_stim_right = "opto_stims/example_opto_stim2.mat"
###################################################

###################################################
### Only edit this section when switching rigs  ###
###################################################

cams = []
ao_trigger = []
ai_audio = []
ai_exposure = []
ao_opto = []
ai_opto_loopback = []
opto_stim = []

if record_left_camera:
    # MurthyLab-PC05 -> Cam1
    cams.append("16276625")
    ao_trigger.append("Dev1/ao0")
    ai_audio.append("Dev1/ai0:8")
    ai_exposure.append("Dev1/ai15")
    ao_opto.append("Dev1/ao1")
    ai_opto_loopback.append("Dev1/ai9")
    opto_stim.append(opto_stim_left)

if record_right_camera:
    # MurthyLab-PC05 -> Cam2
    cams.append("18159111")
    ao_trigger.append("Dev1/ao2")
    ai_audio.append("Dev1/ai16:24")
    ai_exposure.append("Dev1/ai31")
    ao_opto.append("Dev1/ao3")
    ai_opto_loopback.append("Dev1/ai25")
    opto_stim.append(opto_stim_right)
###################################################

###################################################
############### Do not edit below!! ###############
###################################################


import leap_rigs
import datetime
import time
import numpy as np
import glob
import os
import h5py

# Setup session naming
expt_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
data_path = f"D:/Motif/daq/daq.{expt_name}.h5"

# Get session metadata from motif
# metadata = {"title": expt_name, "description": hostname}
metadata = {"title": expt_name}
metadata.update(leap_rigs.motif.get_experiment_metadata())

# Acquire motif controller
motif = leap_rigs.motif.get_motif_remote()

# Start cameras
for cam in cams:
    cam_filename = f"{expt_name}_{cam}"
    motif.call(f"camera/{cam}/recording/start", codec="h264-gpu", filename=cam_filename, metadata=metadata)
    print(f"Starting recording with camera {cam} -> {cam_filename}")


# Setup DAQ
daq_controller = leap_rigs.daq.DAQController(
    ao_trigger=ao_trigger,
    ai_audio=ai_audio,
    ai_exposure=ai_exposure,
    ao_opto=ao_opto,
    ai_opto_loopback=ai_opto_loopback,
    data_path=data_path,
    opto_data=opto_stim,
    daq_sample_frequency=daq_sample_frequency,
    cam_trigger_frequency=cam_trigger_frequency,
    callback_sample_frequency=callback_sample_frequency,
    expected_duration=experiment_duration + 1,
)

# Setup DAQ
daq_controller.setup_daq()
daq_controller.setup_saving()
print("Setup DAQ and saving")

# Start DAQ and triggering after a delay
daq_controller.start_saving()
print("Started saving")
time.sleep(2.5)
daq_controller.start_triggering()
print("Started triggering")


# Keep checking if experiment is done...
t0 = time.time()
done = False
while not done:
    # Check if we're past the max duration
    time_elapsed = time.time() - t0
    max_duration_expired = time_elapsed > (experiment_duration * 60)

    # Check if no cameras are running
    still_recording = any([motif.is_recording(cam) for cam in cams])
    
    # Determine if we're done
    done = max_duration_expired or (not still_recording)

    # Pause
    if not done:
        time.sleep(5.0)
        print(f"[t = {time_elapsed / 60:.2f} min] Still recording")

total_duration = time.time() - t0
print("Stopping experiment after %.1f minutes" % (total_duration / 60))

# Stop triggering
daq_controller.stop_triggering()
print("Stopped triggering")

# Send stop signal to cameras
for cam in cams:
    motif.call(f"camera/{cam}/recording/stop")
    print(f"Stopping camera: {cam}")
    time.sleep(1)

# Wait for them to finish
done_recording = False
while not done_recording:
    print("Waiting for all cameras to finish recording...")

    # Check with Motif if cameras are running
    still_recording = []
    for cam in cams:
        try:
            cam_is_recording = motif.is_recording(cam)
        except:
            print(f"Failed to check if {cam} is recording! Will assume it is stopped.")
            cam_is_recording = False
        if cam_is_recording:
            print(f"Waiting for {cam} to finish recording...")
        still_recording.append(cam_is_recording)
        time.sleep(1)
    done_recording = not any(still_recording)

    if not done_recording:
        time.sleep(1)

time.sleep(2.5)

# Stop saving DAQ now that cameras are done
daq_controller.stop_saving()
print("Stopped saving.")

# Force close all DAQ tasks
# TODO: Figure out why they don't get closed in the above commands.
print("Making sure all tasks are closed...")
daq_controller.close_all_tasks()

print("Tasks closed:")
daq_controller.check_tasks()

# Pull out data from the temporary DAQ HDF5 data file into session folders
print("Temporary data_path:", data_path)
final_data_paths = []
if daq_controller.is_saving:
    time.sleep(3)

    # Move data to final session folder
    with h5py.File(data_path, "r") as daqF:
        daq_data = daqF["data"]
        has_opto = [stim is not None for stim in opto_stim]
        for c, (cam, cam_has_opto) in enumerate(zip(cams, has_opto)):
            cam_filename = f"{expt_name}_{cam}"
            vid_dst = f"D:/Motif/{cam_filename}"
            print(f"Saving camera data to: {vid_dst}")
            for _ in range(3):
                vid_src = glob.glob(f"D:/Motif/{cam}/{cam_filename}*")[0]
                print(f"Trying to move: {vid_src} -> {vid_dst}")

                try:
                    os.rename(vid_src, vid_dst)
                    print(f"Moved: {vid_src} -> {vid_dst}")
                    break
                except:
                    time.sleep(3)

            try:
                vid_daq_path = f"{vid_dst}/daq.h5"
                with h5py.File(vid_daq_path, "w") as f:
                    audio_inds = [daq_controller.channel_map.index(f"audio{i}.cam{c}") for i in range(9)]
                    f.create_dataset("audio", data=daq_data[audio_inds, :], compression="gzip", compression_opts=1)
                    print(f"Saved audio (data indices: {audio_inds})")

                    exposure_ind = daq_controller.channel_map.index(f"exposure.cam{c}")
                    f.create_dataset("sync", data=daq_data[exposure_ind, :], compression="gzip", compression_opts=1)
                    print(f"Saved sync (data index: {exposure_ind})")

                    if cam_has_opto:
                        opto_loopback_ind = daq_controller.channel_map.index(f"opto_loopback.cam{c}")
                        f.create_dataset("opto", data=daq_data[opto_loopback_ind, :], compression="gzip", compression_opts=1)
                        print(f"Saved opto outputs (data index: {opto_loopback_ind})")

                print(f"Saved DAQ data to final session folder: {vid_daq_path}")
                final_data_paths.append(vid_dst)
            except:
                print(f"Failed to save DAQ data to final session folder: {vid_daq_path}")

    # Delete temporary DAQ HDF5 data file
    try:
        os.remove(data_path)
        print("Deleted temporary data_path:", data_path)
    except:
        print("Failed to delete temporary data_path:", data_path)

time.sleep(3)
print("Session finished.")
if len(final_data_paths) > 0:
    print("\nFinal data_paths:")
    print("\n".join(final_data_paths))
