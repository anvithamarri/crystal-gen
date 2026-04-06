import random
import zmq
import numpy as np
from pymatgen.core import Structure
from chgnet.model import CHGNet  # FIX: Correct import
from chgnet.model.dynamics import CHGNetCalculator  # ADD: For relaxation
import numpy as np

class CHGNetScorer:
    def __init__(self):
        # FIX 1: Load model correctly
        self.model = CHGNet.load()
        # FIX 2: Remove Streamlit dependency - scorer.py shouldn't depend on Streamlit
        print("CHGNet Loaded: Ready for Stability Prediction")
        # NOTE: Removed st.info() - this is a library, not a UI component

    def score(self, cif_string: str) -> float:
        try:
            # FIX 3: Convert CIF to Structure
            struct = Structure.from_str(cif_string, fmt="cif")
            
            # FIX 4: Use correct API for predictions
            # predict_structure returns dict with 'e', 'f', 's' keys
            prediction = self.model.predict_structure(struct)
            
            # FIX 5: Access correct key - 'e' for total energy, not 'energy'
            energy = prediction['e']
            
            # FIX 6: Correct reward logic
            # Lower (more negative) energy = more stable = higher reward
            reward = -energy
            
            return float(reward)
            
        except Exception as e:
            # FIX 7: Better error handling
            print(f"CHGNetScorer error: {e}")
            return -10.0


class CIFScorer:
    """
    An abstract CIF scorer. A scorer provides a heuristic score for a completed CIF.
    """
    def score(self, cif: str) -> float:
        """
        Returns a score for the CIF. A higher score is better (max 0.0).
        """
        pass

class HeuristicPhysicalScorer(CIFScorer):
    """
    Advanced scorer that rewards realistic density, cubic-like cells, 
    and appropriate atomic spacing.
    """
    def __init__(self, target_density: float = 2.16):
        self.target_rho = target_density

    def score(self, cif: str) -> float:
        try:
            struct = Structure.from_str(cif, fmt="cif")
            
            # 1. Density (Target 2.16)
            rho_error = abs(struct.density - self.target_rho)
            
            # 2. Cubic Shape (a=b=c) - Increased weight to 15.0
            a, b, c = struct.lattice.abc
            shape_error = (abs(a - b) + abs(b - c) + abs(a - c)) * 15.0
            
            # 3. Angle Penalty (Alpha, Beta, Gamma should be 90)
            angles = struct.lattice.angles
            angle_error = sum([abs(ang - 90.0) for ang in angles]) * 2.0
            
            # 4. Axis Penalty (Max length)
            axis_penalty = 20.0 if max(a, b, c) > 7.0 else 0.0

            # 5. Min Distance Check
            min_dist = struct.get_all_neighbors(1.0)[0][0].nn_distance if len(struct) > 1 else 0
            dist_penalty = 10.0 if min_dist < 2.0 else 0.0
            
            # 6. Symmetry Bonus (Fm-3m / No. 225)
            sym_bonus = 10.0 if struct.get_space_group_info()[1] == 225 else 0.0

            return -(rho_error + shape_error + angle_error + axis_penalty + dist_penalty) + sym_bonus
        except:
            return -100.0


class ZMQScorer(CIFScorer):
    """
    Remains here for when you resolve the ALIGNN/DGL installation.
    Connects to an external server (zmq_server.py).
    """
    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 30000):
        print(f"ZeroMQ CIFScorer connecting to {host}:{port}")
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{port}")

    def score(self, cif: str) -> float:
        try:
            self._socket.send_string(cif)
            message = self._socket.recv_string()
            return float(message)
        except Exception:
            return -10.0

class RandomScorer(CIFScorer):
    """
    Simple random baseline for debugging.
    """
    def __init__(self, min_score: float = -5., max_score: float = 5., seed: int = None):
        self._local_random = random.Random(seed)
        self._min_score = min_score
        self._max_score = max_score

    def score(self, cif: str) -> float:
        return self._local_random.uniform(self._min_score, self._max_score)
