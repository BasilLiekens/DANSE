import numpy as np
import os
import pyroomacoustics as pra
import resampy
import scipy.signal as signal
import scipy.io.wavfile as wav

from .asc import AcousticScenario, Parameters


def generate_RIRs(p: Parameters) -> tuple[np.ndarray, np.ndarray, pra.ShoeBox]:
    """
    Wrapper method around the "AcousticScenario" class in "asc.py" which returns
    the room object. This method just extracts these RIRs and either pads or
    truncates them to have the correct length. Function is based on the
    "main.py" script by P. Didier in the "pra_demo" repo.

    This function assumes all nodes have the same number of mics.

    Parameters
    -----------------
        p: Parameters
            The parameters that were loaded from the cfg.yml file.

    Returns
    -----------------
        tuple containing two np.ndarrays, the first containing the RIRs from the
        audio sources to all mics, the second one from all noise sources to all
        mics. Shape ["RIR_length" x "nb Mics" x "nb Sources"]. Besides the room
        that resulted in these RIRs is also included for further reference.
    """
    RIRs_audio = np.zeros([p.RIR_length, p.Mk * p.K, p.Ns])
    RIRs_noise = np.zeros([p.RIR_length, p.Mk * p.K, p.Nn])

    room = AcousticScenario(p).generate_room()
    if room.rir is not None:
        audioRIRs = list(map(lambda x: x[:][: p.Ns], room.rir))
        noiseRIRs = list(map(lambda x: x[:][p.Ns :], room.rir))

    else:
        raise ValueError("Rirs were empty!")

    for i in range(p.Mk * p.K):
        for j in range(p.Ns):
            RIR_len = len(audioRIRs[i][j])
            if RIR_len > p.RIR_length:
                print("/!\\ truncating RIR /!\\")
                RIRs_audio[:, i, j] = audioRIRs[i][j][: p.RIR_length]
            else:
                RIRs_audio[:RIR_len, i, j] = audioRIRs[i][j]

    for i in range(p.Mk * p.K):
        for j in range(p.Nn):
            RIR_len = len(noiseRIRs[i][j])
            if RIR_len > p.RIR_length:
                print("/!\\ truncating RIR /!\\")
                RIRs_noise[:, i, j] = noiseRIRs[i][j][: p.RIR_length]
            else:
                RIRs_noise[:RIR_len, i, j] = noiseRIRs[i][j]

    return RIRs_audio, RIRs_noise, room


def create_micsigs(p: Parameters) -> tuple[np.ndarray, np.ndarray, pra.ShoeBox]:
    """
    Given the path to the config file with all data for the room and which input
    signals to use for the generation, generate the microphone signals by first
    generating the RIRs and subsequently convolving them with the audio signals.
    This functions returns the contribution of each individual microphone to
    each individual mic to be able to extract VAD and groundtruth signals
    properly.

    Parameters
    ------------
        p: Parameters
            The parameters, loaded from the yaml file

    Returns
    ------------
        Two numpy arrays, one with the signal pure signal component and one with
        all noise contributions (both spatially correlated and measurement
        noise). The shape of the arrays is
        ["nb Mics" x "signal_length" x "nb Sources"]. Besides, the room used to
        generate the RIRs is also returned for further reference.
    """
    # sampling frequencies per node, use the lowest frequency as "base" to
    # ensure every node has enough samples: it will require a longer recording
    # for the same amount of samples.
    SRO_Hz = [p.SRO[x] * p.fs / 1e6 for x in range(p.K)]
    nodeFs = [p.fs + SRO_Hz[x] for x in range(p.K)]

    recLen = int(p.signal_length * np.min(nodeFs))
    finalLen = int(p.fs * p.signal_length)

    # load audio and noise, resample if necessary (to the base frequency!),
    # rescale each source signal to unit power
    if p.recorded_signals:
        if len(p.audio_sources) != p.Ns:
            raise ValueError(
                "The number of audio sources should be equal to the number of audio recordings passed for those sources"
            )
        audio_sigs = np.zeros((recLen, p.Ns))

        for i in range(p.Ns):
            fs, data = wav.read(os.path.join(p.audio_base, p.audio_sources[i]))
            if fs != p.fs:
                data = resampy.resample(data, sr_orig=fs, sr_new=p.fs)

            audio_sigs[:, i] = data[:recLen]
    else:
        audio_sigs = 0.1 * np.random.randn(recLen, p.Ns)

    if p.recorded_noise:
        if len(p.noise_sources) != p.Nn:
            raise ValueError(
                "The number of noise sources should be equal to the number of noise recordings passed for those sources"
            )

        noise_sigs = np.zeros((recLen, p.Nn))

        for i in range(p.Nn):
            fs, data = wav.read(os.path.join(p.noise_base, p.noise_sources[i]))
            if fs != p.fs:
                data = resampy.resample(data, sr_orig=fs, sr_new=p.fs)

            noise_sigs[:, i] = data[:recLen]
    else:
        noise_sigs = 0.1 * np.random.randn(recLen, p.Nn)

    # normalize to unit power
    audio_sigs = audio_sigs / np.std(audio_sigs, axis=0, keepdims=True)
    noise_sigs = noise_sigs / np.std(noise_sigs, axis=0, keepdims=True)

    # force audio signal to 0 if "include_silences" is active (to have a more
    # even balance between noise-only and audio + noise segments). Do so here
    # already to have a more natural sound (the full RIR has an influence still)
    if p.include_silences:
        nPeriods = int(np.ceil(audio_sigs.shape[0] / p.silence_period))
        elementVAD = np.ones(p.silence_period, dtype=bool)
        elementVAD[int(p.silence_period * p.silence_duty_cycle) :] = 0
        totalVAD = np.tile(elementVAD, reps=nPeriods)[: audio_sigs.shape[0]]
        audio_sigs[~totalVAD, :] = 0

    # convolve with RIRs at the base frequency
    RIRs_audio, RIRs_noise, room = generate_RIRs(p)

    audioBase = np.zeros((p.Mk * p.K, recLen, p.Ns))
    noiseBase = np.zeros((p.Mk * p.K, recLen, p.Nn))

    for i in range(p.Ns):
        for j in range(RIRs_audio.shape[1]):
            audioBase[j, :, i] = signal.fftconvolve(
                audio_sigs[:, i], RIRs_audio[:, j, i]
            )[: audioBase.shape[1]]

    for i in range(p.Nn):
        for j in range(RIRs_noise.shape[1]):
            noiseBase[j, :, i] += signal.fftconvolve(
                noise_sigs[:, i], RIRs_noise[:, j, i]
            )[: audioBase.shape[1]]

    # resample each node to their own sampling frequency (to simulate SRO)
    audio = np.zeros((p.Mk * p.K, finalLen, p.Ns))
    noise = np.zeros((p.Mk * p.K, finalLen, p.Nn))
    for k in range(p.K):
        lB = k * p.Mk
        uB = (k + 1) * p.Mk
        audioTmp = resampy.resample(
            audioBase[lB:uB, :, :], sr_orig=p.fs, sr_new=nodeFs[k], axis=1
        )

        noiseTmp = resampy.resample(
            noiseBase[lB:uB, :, :], sr_orig=p.fs, sr_new=nodeFs[k], axis=1
        )
        audio[lB:uB, :, :] = audioTmp[:, :finalLen, :]
        noise[lB:uB, :, :] = noiseTmp[:, :finalLen, :]

    # scale the interferers to obtain the desired INR
    desiredPower = np.var(np.sum(audio, axis=2)[0, :]) * 10 ** (-p.SIR / 10)
    scaleFactor = np.sqrt(desiredPower / np.var(np.sum(noise, axis=2)[0, :]))
    noise *= scaleFactor

    if p.measurement_noise:
        sigPower = np.mean(
            np.square(np.sum(audio, axis=2) + np.sum(noise, axis=2)), axis=1
        )
        measNoisePower = sigPower * 10 ** (-p.measurement_SNR / 10)
        measNoise = np.sqrt(
            measNoisePower.reshape((-1, 1, 1)) / noiseBase.shape[2]
        ) * np.random.randn(
            *noiseBase.shape  # use "*" to unpack the tuple returned by ".shape"
        )
        noise += measNoise

    return audio, noise, room
