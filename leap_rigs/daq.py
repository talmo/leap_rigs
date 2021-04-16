"""Utilities for controlling the NI DAQ."""

import numpy as np
import h5py
from time import perf_counter
from typing import Optional, Any, Union, Callable

import motifapi
from motifapi import MotifApi, MotifError

import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration


def make_multichan_trigger_task(
    channels, freq, duty=0.5, low=0, high=3, auto_start=True, task=None
) -> nidaqmx.Task:
    """Make a multi-channel triggering task."""
    if task is None:
        task = nidaqmx.Task()

    for chan in channels:
        task.ao_channels.add_ao_voltage_chan(chan)

    # I could be generic here and calculate the number of samples to achieve perfect
    # duty cycle accuracy, but lets just use a large number instead
    N_SAMPS = 1000
    task.timing.cfg_samp_clk_timing(
        rate=int(N_SAMPS * freq),
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=N_SAMPS,
    )
    writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(
        task.out_stream, auto_start=auto_start
    )
    _nhigh = int(duty * N_SAMPS)

    _samples = np.append(high * np.ones(_nhigh), low * np.zeros(N_SAMPS - _nhigh))
    samples = np.vstack([_samples for _ in channels])

    writer.write_many_sample(samples)

    return task


def make_independent_trigger_task(
    chan, freq, duty=0.5, low=0, high=3, auto_start=True, task=None
) -> nidaqmx.Task:
    """Make an independent (single channel) triggering task."""
    return make_multichan_trigger_task([chan], freq, duty, low, high, auto_start, task)


def make_constant_value_task(v, channel, auto_start=False, task=None) -> nidaqmx.Task:
    """Make a constant value (simple) task."""
    if task is None:
        task = nidaqmx.Task()

    task.ao_channels.add_ao_voltage_chan(channel)
    task.write(v)

    if auto_start:
        task.start()

    return task


class DAQController:
    """NI DAQ controller responsible for setting up inputs, outputs and saving.

    Attributes:
        ao_trigger: Analog output channel for camera triggering.
        ai_audio: Analog input channels for microphones.
        ao_opto: Analog output channel for opto stim. If None (the default), no opto
            stimulation is written out.
        ai_opto_loopback: Analog input channel for opto stim loopback. If None (the
            default), no opto stimulation data is written out.
        data_path: Path to .h5 file that the data will be written out to. If None, no
            saving is performed.
        opto_data: Data to be used for opto stimulation. This can be provided as a numpy
            vector of voltages of any length (will be indexed circularly), a MAT file
            containing this vector in a variable named "stim", or a callable signal
            generation function that returns a vector or scalar float of data. See notes
            for the inputs this function may receive.
        daq_sample_frequency: Sampling rate to set the DAQ to in samples/second.
            Defaults to 10000.
        cam_trigger_frequency: Sampling rate to trigger the camera at in frames/second.
            Defaults to 150.
        callback_sample_frequency: Frequency (in samples) of data reading and opto stim.
            This determines the opto latency for closed-loop experiments, but has
            basically no effect for open-loop. Defaults to 250.
        expected_duration: Expected duration of the session in minutes. This is used to
            preallocate the output HDF5 dataset for writing. If stopped before this
            duration, the dataset will be cropped to the actual length, so just set this
            to a comfortably larger value than expected. Defaults to 35.

    Notes:
        When the opto_data attribute is specified as a function, this serves as a signal
        generator. The function will receive some metadata as input as is expected to
        return a numpy vector of voltages, or a scalar float.

        The function will receive these inputs:
            s0: Starting sample index for the chunk.
            s1: Ending sample index for the chunk.
            number_of_samples: Number of samples in the chunk.
            chunk_input_data: Data read in from the analog input channels.
            daq: The DAQController that the function is being called from.

        Example:
            def opto_stim(s0, s1, number_of_samples, chunk_input_data, daq):
                # Turn on opto stim every other second.
                if (s0 // daq.Fs) % 2 == 0:
                    return 0.0
                else:
                    return 3.0
    """

    def __init__(
        self,
        ao_trigger: str,
        ai_audio: str,
        ao_opto: Optional[str] = None,
        ai_opto_loopback: Optional[str] = None,
        data_path: Optional[str] = None,
        opto_data: Optional[Union[str, np.ndarray, Callable]] = None,
        daq_sample_frequency: int = 10000,
        cam_trigger_frequency: int = 150,
        callback_sample_frequency: int = 250,
        expected_duration: float = 35,
    ):

        self.ao_trigger = ao_trigger
        self.ai_audio = ai_audio
        self.ai_exposure = ai_exposure
        self.ao_opto = ao_opto
        self.ai_opto_loopback = ai_opto_loopback
        self.data_path = data_path
        self.daq_sample_frequency = daq_sample_frequency
        self.cam_trigger_frequency = cam_trigger_frequency
        self.callback_sample_frequency = callback_sample_frequency
        self.expected_duration = expected_duration

        self.read_task = None
        self.trigger_task = None
        self.opto_task = None
        self._f = None
        self._ds_data = None
        self.sample_idx = 0

        self.opto_data = opto_data
        if type(opto_data) == str:
            if opto_data.endswith(".mat"):
                from scipy.io import loadmat

                self.opto_data = loadmat(opto_data)["stim"].squeeze().astype("float64")
                print(f"Loaded opto_data ({opto_data}): {len(self.opto_data)} samples")
            else:
                raise ValueError("Paths to opto stimulation data must end in '.mat'.")

    @property
    def Fs(self) -> int:
        """Return the sampling frequency of the DAQ. Alias for daq_sample_frequency."""
        return self.daq_sample_frequency

    @property
    def is_saving(self) -> bool:
        """Return whether data saving is enabled."""
        return self.data_path is not None

    @property
    def is_writing_opto(self) -> bool:
        """Return whether opto stim writing is enabled."""
        return self.ao_opto is not None

    @property
    def is_reading_opto(self) -> bool:
        """Return whether opto loopback reading is enabled."""
        return self.ai_opto_loopback is not None

    def setup_daq(self):
        """Set up DAQ tasks for reading and writing."""
        if self.read_task is not None:
            return

        self.read_task = nidaqmx.Task()

        # Audio channels
        self.read_task.ai_channels.add_ai_voltage_chan(
            self.ai_audio,
            min_val=-10.0,
            max_val=10.0,
            terminal_config=TerminalConfiguration.NRSE,
        )
        print(f"Added audio input channels: {self.ai_audio}")

        # Camera exposure readout
        self.read_task.ai_channels.add_ai_voltage_chan(
            self.ai_exposure, min_val=0.0, max_val=5.0
        )
        print(f"Added camera exposure input channel: {self.ai_exposure}")

        if self.is_reading_opto:
            # Opto loopback
            self.read_task.ai_channels.add_ai_voltage_chan(
                self.ai_opto_loopback, min_val=0.0, max_val=10.0
            )
            print(f"Added opto loopback input channel: {self.ai_opto_loopback}")

        # Set sampling rate on the DAQ
        self.read_task.timing.cfg_samp_clk_timing(
            rate=self.daq_sample_frequency, sample_mode=AcquisitionType.CONTINUOUS
        )
        print(f"Set DAQ sampling rate: {self.daq_sample_frequency}")

        # Register read callback
        self.read_task.register_every_n_samples_acquired_into_buffer_event(
            self.callback_sample_frequency, self.callback
        )
        print(f"Setup DAQ callback every {self.callback_sample_frequency} samples")

        # Camera triggering
        self.trigger_task = make_independent_trigger_task(
            self.ao_trigger, self.cam_trigger_frequency, duty=0.1, auto_start=False
        )
        print(
            f"Added camera triggering channel: {self.ao_trigger} at {self.cam_trigger_frequency} FPS"
        )

        # Opto stim
        if self.is_writing_opto:
            self.opto_task = make_constant_value_task(0.0, self.ao_opto)
            print(f"Added opto triggering channel: {self.ao_opto}")

    @property
    def n_input_channels(self) -> int:
        """Return the number of analog input channels."""
        return len(self.read_task.ai_channels.channel_names)

    def setup_saving(self):
        """Setup HDF5 output file for writing if saving is enabled."""
        if not self.is_saving:
            return

        n_expected_samples = self.Fs * 60 * self.expected_duration
        self._f = h5py.File(self.data_path, "w")
        self._ds_data = self._f.create_dataset(
            "data",
            (self.n_input_channels, n_expected_samples),
            maxshape=(self.n_input_channels, None),
            dtype=np.dtype("float64"),
            chunks=True,
            compression="gzip",
            compression_opts=1,
        )
        print(f"Created HDF5 data file: {self.data_path}")

    def callback(
        self,
        task_handle: nidaqmx.Task,
        every_n_samples_event_type: Any,
        number_of_samples: int,
        callback_data: Any,
    ):
        """Callback function that is triggered every N samples.

        Do not call this function directly -- this will be called by the nidaqmx API.

        Args:
            task_handle: The nidaqmx.Task associated with this callback. This is
                registered in setup_daq with read_task.
            every_n_samples_event_type: N/A
            number_of_samples: Number of samples since the last call.
            callback_data: N/A
        """
        # Calculate sample and time (in secs) of the current chunk
        s0, s1 = self.sample_idx, self.sample_idx + number_of_samples
        t0, t1 = s0 / self.Fs, s1 / self.Fs

        # Get data from task inputs.
        chunk_input_data = np.asarray(self.read_task.read(number_of_samples))
        if self.save:
            # Write to HDF5.
            self._ds_data[
                :, self.sample_idx : (self.sample_idx + number_of_samples)
            ] = chunk_input_data

        # Update the row (sample) that we're on.
        self.sample_idx += number_of_samples

        if self.opto_task is None:
            # No opto task, so just return early.
            return 0

        # Get the stimulation data.
        if callable(self.opto_data):
            stim = self.opto_data(
                s0=s0,
                s1=s1,
                number_of_samples=number_of_samples,
                chunk_input_data=chunk_input_data,
                daq=self,
            )
        else:
            stim = self.opto_data

        if stim is None:
            # No stim, so just do nothing.
            return 0

        if not np.isscalar(stim) and len(stim) != number_of_samples:
            # Index into a stimulus vector (wraps around if vector is too short)
            next_s0, next_s1 = s1, s1 + number_of_samples
            stim = np.take(self.opto_data, range(next_s0, next_s1), mode="wrap")

        # Write stim to opto channel.
        self.opto_task.write(stim, auto_start=True)

        return 0

    def start(self):
        """Setup and start the session."""
        self.setup_daq()
        self.setup_saving()
        self.read_task.start()

    def turn_off_opto(self):
        """Send a zero voltage pulse to the opto output to turn off the LED."""
        if self.opto_task is not None:
            self.opto_task.write(0.0, auto_start=True)

    def stop(self):
        """Stop opto and clean up the saved output."""
        self.turn_off_opto()
        if self.is_saving:
            self._ds_data.resize((self.n_input_channels, self.sample_idx))
            self._f.close()


def test_opto_stim_fn(s0, s1, number_of_samples, chunk_input_data, daq):
    # Turn on opto stim every other second.
    if (s0 // daq.Fs) % 2 == 0:
        return 0.0
    else:
        return 3.0