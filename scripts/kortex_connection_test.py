#!/usr/bin/env python3
"""Kortex connection test for Kinova Gen3.

Reads every piece of state the recorder will log at 30 Hz, prints it once,
then measures actual achievable read rate.

Run inside the kinova_env venv:
    source ~/kinova_env/bin/activate
    python3 kortex_connection_test.py

Does NOT move the arm. Read-only.
"""

import argparse
import math
import time

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2

DEFAULT_IP = "192.168.1.10"
DEFAULT_USER = "admin"
DEFAULT_PASS = "admin"
TCP_PORT = 10000


class DeviceConnection:
    """Context manager for a Kortex TCP session."""

    def __init__(self, ip, port=TCP_PORT, username=DEFAULT_USER, password=DEFAULT_PASS):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.transport = None
        self.router = None
        self.session_manager = None

    def __enter__(self):
        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)
        self.transport.connect(self.ip, self.port)

        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self.username
        session_info.password = self.password
        session_info.session_inactivity_timeout = 60000       # ms
        session_info.connection_inactivity_timeout = 2000     # ms

        self.session_manager = SessionManager(self.router)
        self.session_manager.CreateSession(session_info)
        return self.router

    def __exit__(self, exc_type, exc_value, traceback):
        if self.session_manager is not None:
            opts = RouterClientSendOptions()
            opts.timeout_ms = 1000
            self.session_manager.CloseSession(opts)
        if self.transport is not None:
            self.transport.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--username", default=DEFAULT_USER)
    ap.add_argument("--password", default=DEFAULT_PASS)
    ap.add_argument("--rate-samples", type=int, default=100,
                    help="How many reads to time for the rate check.")
    args = ap.parse_args()

    with DeviceConnection(args.ip, username=args.username, password=args.password) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        # ---- Device info -------------------------------------------------
        print("=" * 62)
        print("DEVICE")
        print("=" * 62)
        actuator_count = base.GetActuatorCount().count
        print(f"  actuators           : {actuator_count}")

        try:
            for d in base.GetAllConnectedDevices().device_info:
                print(f"  device              : {d.name}  (fw {d.device_identifier})")
        except Exception as e:
            print(f"  device list unavailable: {e}")

        # ---- One full state read ----------------------------------------
        fb = base_cyclic.RefreshFeedback()

        print()
        print("=" * 62)
        print("JOINTS  (Kortex reports DEGREES)")
        print("=" * 62)
        pos_deg, vel_deg, torque = [], [], []
        for i, act in enumerate(fb.actuators):
            pos_deg.append(act.position)
            vel_deg.append(act.velocity)
            torque.append(act.torque)
            print(f"  j{i}  pos {act.position:9.3f} deg   "
                  f"vel {act.velocity:8.3f} deg/s   tau {act.torque:8.3f} Nm")

        print()
        print("  position (rad) :", [round(math.radians(p), 5) for p in pos_deg])
        print("  velocity (rad/s):", [round(math.radians(v), 5) for v in vel_deg])

        # ---- End-effector pose -------------------------------------------
        print()
        print("=" * 62)
        print("END-EFFECTOR")
        print("=" * 62)
        b = fb.base
        print(f"  position (m)  x={b.tool_pose_x:8.4f}  y={b.tool_pose_y:8.4f}  z={b.tool_pose_z:8.4f}")
        print(f"  orient (deg) rx={b.tool_pose_theta_x:9.3f} "
              f"ry={b.tool_pose_theta_y:9.3f} rz={b.tool_pose_theta_z:9.3f}")
        print()
        print("  NOTE: these Euler angles use ZYX convention in DEGREES and wrap at +/-180.")
        print("        Convert to a rotation matrix or quaternion before computing deltas,")
        print("        or you WILL get discontinuities that corrupt the dataset.")

        # ---- Gripper ------------------------------------------------------
        print()
        print("=" * 62)
        print("GRIPPER")
        print("=" * 62)
        if len(fb.interconnect.gripper_feedback.motor) > 0:
            m = fb.interconnect.gripper_feedback.motor[0]
            print(f"  raw position  : {m.position:.3f}   (0..100 scale)")
            print(f"  normalized    : {m.position / 100.0:.4f}   (0.0 open -> 1.0 closed)")
            print(f"  velocity      : {m.velocity:.3f}")
            print(f"  current (mA)  : {m.current_motor:.1f}")
        else:
            print("  No gripper feedback found. Is the Robotiq 2F-85 attached and configured?")

        # ---- Achievable read rate ----------------------------------------
        print()
        print("=" * 62)
        print(f"READ RATE  ({args.rate_samples} samples)")
        print("=" * 62)
        t0 = time.perf_counter()
        for _ in range(args.rate_samples):
            base_cyclic.RefreshFeedback()
        elapsed = time.perf_counter() - t0
        per_read_ms = 1000.0 * elapsed / args.rate_samples
        print(f"  mean per read : {per_read_ms:.2f} ms")
        print(f"  max rate      : {1000.0 / per_read_ms:.1f} Hz")
        if per_read_ms > 33.3:
            print("  WARNING: slower than 30 Hz. Recording at 30 Hz will drop frames.")
        else:
            print("  OK for 30 Hz recording.")

        # ---- Home pose snippet -------------------------------------------
        print()
        print("=" * 62)
        print("HOME POSE  (paste into your recorder config)")
        print("=" * 62)
        print("HOME_JOINTS_DEG = [")
        for p in pos_deg:
            print(f"    {p:.3f},")
        print("]")
        print()
        print("Jog the arm to your chosen start configuration, re-run this,")
        print("and use the values above. Every episode starts and ends here.")


if __name__ == "__main__":
    main()
