# Contents of package:
# Classes and functions related to the acoustic scenario.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import time

import numpy as np
import pyroomacoustics as pra

from .setup import Parameters
from dataclasses import dataclass


@dataclass
class AcousticScenario:
    cfg: Parameters

    def generate_room(self, plotRoom=False):
        """
        Generate an acoustic scenario.

        Parameters
        ----------
        plotRoom : bool, optional
            Whether to plot the room, by default False.

        Returns
        -------
        room: pra.ShoeBox
            The generated room.
        """
        # Create pyroomacoustics room with appropriate material for desired T60
        if self.cfg.t60 == 0:
            maxOrd, eAbs = 0, 0.5  # <-- arbitrary
        else:
            eAbs, maxOrd = pra.inverse_sabine(self.cfg.t60, self.cfg.rd)

        room = pra.ShoeBox(
            p=self.cfg.rd,
            fs=self.cfg.fs,
            max_order=maxOrd,
            air_absorption=False,
            materials=pra.Material(eAbs),
            temperature=self.cfg.temperature,
            humidity=self.cfg.humidity,
        )

        def _rand_pos(wallDist: float = 0.0):
            """
            Returns a random position in the room.

            Parameters
            ----------
            wallDist : float, optional
                Minimum distance from the room walls, by default 0.

            Returns
            -------
            np.ndarray
                The random position.
            """
            return np.random.uniform(wallDist, self.cfg.rd - wallDist, self.cfg.dim)

        def _gen_sensor_pos(nodePos: np.ndarray):
            """
            Generate sensor positions around a given node position.

            Parameters
            ----------
            nodePos : np.ndarray
                The node position.

            Returns
            -------
            sensorPos: np.ndarray
                The sensor positions around the node.
            """

            def __rand_mic_pos():
                return np.random.uniform(
                    -self.cfg.node_diameter / 2,
                    self.cfg.node_diameter / 2,
                    self.cfg.dim,
                )

            maxAttempts = 100
            sensorPos = [None] * self.cfg.Mk
            for m in range(self.cfg.Mk):
                nAttempts = 0
                pos = __rand_mic_pos() + nodePos
                # Make sure the microphone is in the room and not too close to another microphone
                while (
                    np.any(pos < 0)
                    or np.any(pos > self.cfg.rd)
                    or (
                        m > 0
                        and np.min(
                            [np.linalg.norm(pos - sensorPos[m2]) for m2 in range(m)]
                        )
                        < self.cfg.min_inter_sensor_d
                    )
                ):
                    pos = __rand_mic_pos() + nodePos
                    nAttempts += 1
                    if nAttempts >= maxAttempts:
                        raise ValueError(
                            "Could not find a valid microphone position. Try modifying the parameters."
                        )
                sensorPos[m] = pos
            return sensorPos

        # Add desired sources
        for i in range(self.cfg.Ns):
            room.add_source(_rand_pos(self.cfg.min_d_sources_wall))
        # Add noise sources
        for i in range(self.cfg.Nn):
            room.add_source(_rand_pos(self.cfg.min_d_sources_wall))

        # Generate node positions
        sensorPos = np.zeros((self.cfg.K, self.cfg.Mk, self.cfg.dim))
        for k in range(self.cfg.K):
            nodePos = _rand_pos(self.cfg.min_d_nodes_wall)
            # Generate microphone positions
            sensorPos[k, :, :] = _gen_sensor_pos(nodePos)

        # Add microphones to the room
        micArray = pra.MicrophoneArray(
            sensorPos.reshape(-1, sensorPos.shape[-1]).T, room.fs
        )
        room.add_microphone_array(micArray)

        if plotRoom:
            room.plot()

        # Simulate impulse responses
        t0 = time.time()
        room.compute_rir()
        # print(f"RIRs computed in {time.time() - t0:.2f} s.")
        # The RIRs are stored in room.rir as a (`self.cfg.K * self.cfg.Mk`)-list
        # of (`self.cfg.Ns + self.cfg.Nn`)-list of numpy arrays.
        # To retrieve the RIR between the `k`-th node and the `n`-th source, use
        # `room.rir[k][n]`

        return room
