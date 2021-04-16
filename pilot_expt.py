import leap_rigs
import datetime
import time


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
motif.call('camera/%s/recording/start' % cam_sn,
                 codec='h264-gpu',
                 filename=f"{expt_name}_{cam_sn}",
                 metadata=metadata)

##########
stream_poller = leap_rigs.motif.StreamPoller(api=motif_api, camera_sn=cam_sn)
live_predictor = leap_rigs.tracking.LivePredictor.load_model(model_paths, get_image_fn=lambda: stream_poller.latest_image)

stream_poller.start()
live_predictor.start()

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
        return leap_rigs.flies.compute_features(self.last_pose_f, self.last_pose_m)

poses = PoseBuffer()

# opto_stim = leap_rigs.daq.test_opto_stim_fn
def opto_stim(s0, s1, number_of_samples, chunk_input_data, daq):
    (img, meta), (pred, lp_timestamp) = lp.last_data_and_prediction
    if meta is not None:
        frame_idx, vr_timestamp = meta
        latency = lp_timestamp - vr_timestamp

        poses.update(pred)
        feats = poses.compute_features()

        do_trigger = (feats.min_dist < 2) and (np.abs(feats.ang_f_rel_m) < 25)
        msg = f"min_dist = {feats.min_dist:.1f} mm / ang = {feats.ang_f_rel_m:.1f} / trigger: {do_trigger}"
        print(msg)

        if do_trigger:
            return 3.0
    return 0.0
##########

# Setup DAQ
daq_controller = leap_rigs.daq.DAQController(
    ao_trigger=ao_trigger,
    ai_audio=ai_audio,
    ao_opto=ao_opto,
    ai_opto_loopback=ai_opto_loopback,
    data_path=data_path,
    opto_data=opto_stim,
    daq_sample_frequency=daq_sample_frequency,
    cam_trigger_frequency=cam_trigger_frequency,
    callback_sample_frequency=callback_sample_frequency,
    expected_duration=experiment_duration + 1,
)


# Start DAQ and triggering
daq_controller.start()
time.sleep(1.5)


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

total_duration = time.time() - t0
print("Stopping experiment after %.1f minutes" % (total_duration / 60))

# Stop triggering
daq_controller.stop()
print("Stopped triggering")

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

