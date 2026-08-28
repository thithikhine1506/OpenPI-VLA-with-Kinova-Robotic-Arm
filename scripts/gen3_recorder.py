#!/usr/bin/env python3
"""
Kinova Gen3 teleop recorder -- Xbox controller, 15 Hz, two cameras.

Task: "pick up the red block and put it in the white bowl"

Reads /dev/input/js0 directly (no pygame/SDL -- SDL failed to see this
controller even though the kernel device works fine).

Writes one HDF5 per episode into ./episodes/. Stores a SUPERSET of state so
the action space can be chosen at conversion time without re-collecting.

Handles the hazards found during connection testing:
  * Joint angles wrap 0<->360 (j4 sat at 359.78 deg) -> unwrapped continuously
  * EE Euler rx sat at 179.67 deg, ON the wrap boundary -> stored as rotation
    matrix + quaternion, never as raw Euler deltas
  * RefreshFeedback costs ~25 ms -> locked to 15 Hz, overruns reported

Usage:
    python3 gen3_recorder.py --calibrate     # learn button numbers (do once)
    python3 gen3_recorder.py --check         # test everything, no motion
    python3 gen3_recorder.py                 # record
    python3 gen3_recorder.py --no-wrist      # front camera only
"""

import argparse
import dataclasses
import json
import math
import os
import struct
import sys
import threading
import time

import cv2
import h5py
import numpy as np
import pyrealsense2 as rs

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, Session_pb2

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

TASK = "pick up the red block and put it in the white bowl"

ROBOT_IP = "192.168.1.10"
ROBOT_USER = "admin"
ROBOT_PASS = "admin"

# From kortex_connection_test.py. Re-run after jogging to your real start pose.
HOME_JOINTS_DEG = [0.280, 3.684, 184.813, 240.679, 359.782, 304.277, 95.178]

FPS = 15
DT = 1.0 / FPS

JS_DEV = "/dev/input/js0"
PADMAP = "padmap.json"

FRONT_CAM = 0
WRIST_CAM_URL = "rtsp://192.168.1.10/color"
IMG_W, IMG_H = 640, 480

LIN_SPEED = 0.10       # m/s at full deflection -- start slow
ANG_SPEED = 25.0       # deg/s at full deflection
DEADZONE = 0.15
GRIPPER_STEP = 0.04

# VERIFY WITH --check BEFORE COLLECTING.
# Open the gripper fully and read the raw value:
#   open reads ~0   -> False
#   open reads ~100 -> True
GRIPPER_INVERT = False

OUT_DIR = "episodes"

# Axes confirmed from raw device dump -- these are NOT guesses.
AX_LX, AX_LY = 0, 1
AX_RX, AX_RY = 3, 4
AX_LT, AX_RT = 2, 5

CONTROLS = [
    ("A",     "A  (start recording / stop and SAVE)"),
    ("B",     "B  (stop and DISCARD)"),
    ("X",     "X  (close gripper)"),
    ("Y",     "Y  (open gripper)"),
    ("LB",    "LB (roll wrist one way)"),
    ("RB",    "RB (roll wrist other way)"),
    ("HOME",  "the button you want for RETURN TO HOME"),
    ("QUIT",  "the button you want for QUIT"),
]


# ----------------------------------------------------------------------------
# Joystick -- direct /dev/input/jsN reader in a background thread
# ----------------------------------------------------------------------------

class Joystick:
    """Non-blocking joystick state. The kernel js protocol is 8-byte events:
    uint32 time, int16 value, uint8 type, uint8 number."""

    FMT, SZ = "IhBB", 8

    def __init__(self, dev=JS_DEV):
        self.f = open(dev, "rb")
        self.axes = {}
        self.buttons = {}
        self._lock = threading.Lock()
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        time.sleep(0.2)   # let the init burst land

    def _loop(self):
        while not self._stop:
            ev = self.f.read(self.SZ)
            if not ev:
                break
            _, val, typ, num = struct.unpack(self.FMT, ev)
            kind = typ & ~0x80
            with self._lock:
                if kind == 0x01:
                    self.buttons[num] = val
                elif kind == 0x02:
                    self.axes[num] = val / 32767.0

    def axis(self, n):
        with self._lock:
            return self.axes.get(n, 0.0)

    def button(self, n):
        with self._lock:
            return self.buttons.get(n, 0)

    def snapshot_buttons(self):
        with self._lock:
            return dict(self.buttons)

    def close(self):
        self._stop = True
        try:
            self.f.close()
        except Exception:
            pass


def calibrate(js):
    """Walk the user through pressing each control; save the numbers."""
    print("\nButton calibration. Press each control when prompted.\n")
    mapping = {}
    for key, desc in CONTROLS:
        # wait for all buttons released
        while any(js.snapshot_buttons().values()):
            time.sleep(0.05)
        print(f"  Press {desc} ... ", end="", flush=True)
        num = None
        while num is None:
            for n, v in js.snapshot_buttons().items():
                if v:
                    num = n
                    break
            time.sleep(0.02)
        mapping[key] = num
        print(f"button {num}")
        time.sleep(0.3)

    with open(PADMAP, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nSaved to {PADMAP}:")
    for k, v in mapping.items():
        print(f"  {k:5s} -> button {v}")
    return mapping


def load_padmap():
    if not os.path.exists(PADMAP):
        sys.exit(f"No {PADMAP}. Run with --calibrate first.")
    with open(PADMAP) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Rotation math -- matrices and quaternions only, never Euler deltas
# ----------------------------------------------------------------------------

def euler_zyx_deg_to_matrix(rx_deg, ry_deg, rz_deg):
    """Kortex tool_pose_theta_{x,y,z} -> 3x3 rotation matrix (R = Rz@Ry@Rx)."""
    rx, ry, rz = map(math.radians, (rx_deg, ry_deg, rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_quat(R):
    """-> (w,x,y,z), canonical hemisphere so there are no sign flips."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w, x, y, z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w, x, y, z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w, x, y, z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


class JointUnwrapper:
    """Kortex reports continuous joints in [0,360). j4 sat at 359.78 -- one
    nudge from wrapping to 0.x, which would read as a -359.8 deg delta and
    poison the action targets. This tracks and removes the wraps."""

    def __init__(self):
        self.prev = None
        self.offset = None

    def __call__(self, deg):
        deg = np.asarray(deg, dtype=np.float64)
        if self.prev is None:
            self.prev = deg.copy()
            self.offset = np.zeros_like(deg)
            return deg.copy()
        d = deg - self.prev
        self.offset -= 360.0 * (d > 180.0)
        self.offset += 360.0 * (d < -180.0)
        self.prev = deg.copy()
        return deg + self.offset

    def reset(self):
        self.prev = None
        self.offset = None


def dz(v, t=DEADZONE):
    return 0.0 if abs(v) < t else (v - math.copysign(t, v)) / (1.0 - t)


# ----------------------------------------------------------------------------
# Robot
# ----------------------------------------------------------------------------

class DeviceConnection:
    def __init__(self, ip, port=10000, username=ROBOT_USER, password=ROBOT_PASS):
        self.ip, self.port = ip, port
        self.username, self.password = username, password
        self.transport = self.router = self.session_manager = None

    def __enter__(self):
        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)
        self.transport.connect(self.ip, self.port)
        info = Session_pb2.CreateSessionInfo()
        info.username = self.username
        info.password = self.password
        info.session_inactivity_timeout = 60000
        info.connection_inactivity_timeout = 2000
        self.session_manager = SessionManager(self.router)
        self.session_manager.CreateSession(info)
        return self.router

    def __exit__(self, *exc):
        if self.session_manager:
            o = RouterClientSendOptions()
            o.timeout_ms = 1000
            self.session_manager.CloseSession(o)
        if self.transport:
            self.transport.disconnect()


def send_twist(base, lin, ang):
    cmd = Base_pb2.TwistCommand()
    cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    cmd.duration = 0
    cmd.twist.linear_x, cmd.twist.linear_y, cmd.twist.linear_z = lin
    cmd.twist.angular_x, cmd.twist.angular_y, cmd.twist.angular_z = ang
    base.SendTwistCommand(cmd)


def send_gripper(base, value):
    raw = (1.0 - value) if GRIPPER_INVERT else value
    cmd = Base_pb2.GripperCommand()
    cmd.mode = Base_pb2.GRIPPER_POSITION
    f = cmd.gripper.finger.add()
    f.finger_identifier = 1
    f.value = float(np.clip(raw, 0.0, 1.0))
    base.SendGripperCommand(cmd)


def go_home(base, joints_deg):
    action = Base_pb2.Action()
    action.name = "home"
    action.application_data = ""
    for i, ang in enumerate(joints_deg):
        ja = action.reach_joint_angles.joint_angles.joint_angles.add()
        ja.joint_identifier = i
        ja.value = float(ang)
    base.ExecuteAction(action)


# ----------------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------------

class Camera:
    """Intel RealSense D435i colour stream. Plain V4L2 returns black frames on
    this device, so we go through the SDK."""

    def __init__(self, src=None):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
        try:
            self.pipe.start(cfg)
            self.ok = True
        except Exception as e:
            print(f"  RealSense start failed: {e}")
            self.ok = False

    def read(self):
        if not self.ok:
            return None
        try:
            frames = self.pipe.wait_for_frames(timeout_ms=200)
            c = frames.get_color_frame()
            if not c:
                return None
            # .copy() is REQUIRED: asanyarray returns a view into librealsense's
            # internal buffer. Holding views exhausts the 16-frame pool and every
            # later wait_for_frames times out -- episodes cap at exactly 16 frames.
            return np.asanyarray(c.get_data()).copy()
        except Exception:
            return None

    def release(self):
        if self.ok:
            try:
                self.pipe.stop()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Episode
# ----------------------------------------------------------------------------

@dataclasses.dataclass
class Episode:
    joint_pos_rad: list = dataclasses.field(default_factory=list)
    joint_vel_rad: list = dataclasses.field(default_factory=list)
    joint_torque: list = dataclasses.field(default_factory=list)
    ee_pos: list = dataclasses.field(default_factory=list)
    ee_quat: list = dataclasses.field(default_factory=list)
    ee_rotmat: list = dataclasses.field(default_factory=list)
    ee_euler_deg: list = dataclasses.field(default_factory=list)
    gripper: list = dataclasses.field(default_factory=list)
    cmd_twist: list = dataclasses.field(default_factory=list)
    cmd_gripper: list = dataclasses.field(default_factory=list)
    t_mono: list = dataclasses.field(default_factory=list)
    img_front: list = dataclasses.field(default_factory=list)
    img_wrist: list = dataclasses.field(default_factory=list)

    def __len__(self):
        return len(self.t_mono)


def save_episode(ep, path, has_wrist):
    n = len(ep)
    t = np.array(ep.t_mono)
    d = np.diff(t)
    with h5py.File(path, "w") as f:
        f.attrs["task"] = TASK
        f.attrs["fps"] = FPS
        f.attrs["robot_type"] = "kinova_gen3"
        f.attrs["num_frames"] = n
        f.attrs["gripper_convention"] = "0.0=open, 1.0=closed"
        f.attrs["joint_units"] = "radians, unwrapped"
        f.attrs["has_wrist_image"] = has_wrist
        f.attrs["dt_mean_ms"] = float(d.mean()*1000) if n > 1 else 0.0
        f.attrs["dt_max_ms"] = float(d.max()*1000) if n > 1 else 0.0

        for name, arr in [
            ("joint_pos_rad", ep.joint_pos_rad), ("joint_vel_rad", ep.joint_vel_rad),
            ("joint_torque", ep.joint_torque), ("ee_pos", ep.ee_pos),
            ("ee_quat", ep.ee_quat), ("ee_rotmat", ep.ee_rotmat),
            ("ee_euler_deg", ep.ee_euler_deg), ("gripper", ep.gripper),
            ("cmd_twist", ep.cmd_twist), ("cmd_gripper", ep.cmd_gripper),
        ]:
            f.create_dataset(name, data=np.array(arr, np.float32))
        f.create_dataset("t_mono", data=t.astype(np.float64))

        vlen = h5py.special_dtype(vlen=np.uint8)
        df = f.create_dataset("img_front", (n,), dtype=vlen)
        for i, im in enumerate(ep.img_front):
            df[i] = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].flatten()
        if has_wrist:
            dw = f.create_dataset("img_wrist", (n,), dtype=vlen)
            for i, im in enumerate(ep.img_wrist):
                dw[i] = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].flatten()


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=ROBOT_IP)
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    FPS = args.fps
    DT = 1.0 / FPS

    js = Joystick()

    if args.calibrate:
        calibrate(js)
        js.close()
        return

    pad = load_padmap()
    B_A, B_B = pad["A"], pad["B"]
    B_X, B_Y = pad["X"], pad["Y"]
    B_LB, B_RB = pad["LB"], pad["RB"]
    B_HOME, B_QUIT = pad["HOME"], pad["QUIT"]

    has_wrist = not args.no_wrist
    os.makedirs(args.out, exist_ok=True)

    front = Camera(FRONT_CAM)
    print(f"front cam : {'OK' if front.ok else 'FAILED'}")
    wrist = None
    if has_wrist:
        wrist = Camera(WRIST_CAM_URL)
        print(f"wrist cam : {'OK' if wrist.ok else 'FAILED'}  ({WRIST_CAM_URL})")
        if not wrist.ok:
            print("  -> continuing without wrist camera")
            has_wrist = False
    if not front.ok:
        js.close()
        sys.exit("Front camera is mandatory.")

    with DeviceConnection(args.ip) as router:
        base = BaseClient(router)
        cyc = BaseCyclicClient(router)
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(mode)

        unwrap = JointUnwrapper()

        if args.check:
            print("\n--- CHECK: 8 s live readout, no motion, no recording ---")
            t0 = time.perf_counter()
            k = 0
            while time.perf_counter() - t0 < 8.0:
                fb = cyc.RefreshFeedback()
                pos = unwrap([a.position for a in fb.actuators])
                g_raw = fb.interconnect.gripper_feedback.motor[0].position
                g = (100.0-g_raw)/100.0 if GRIPPER_INVERT else g_raw/100.0
                fi = front.read()
                wi = wrist.read() if has_wrist else None
                held = [n for n, v in js.snapshot_buttons().items() if v]
                if k % 5 == 0:
                    print(f"  grip raw={g_raw:6.2f} norm={g:.3f} | "
                          f"rx={fb.base.tool_pose_theta_x:8.2f} | "
                          f"front={'ok' if fi is not None else 'NONE'} "
                          f"wrist={'ok' if wi is not None else '-'} | "
                          f"btn={held} lx={js.axis(AX_LX):+.2f}")
                k += 1
                time.sleep(DT)
            print("\nGRIPPER: open it fully, re-run --check, read raw.")
            print("  raw ~0   -> GRIPPER_INVERT = False")
            print("  raw ~100 -> GRIPPER_INVERT = True")
            front.release()
            if wrist:
                wrist.release()
            js.close()
            return

        print(f"\ntask : {TASK}")
        print(f"rate : {FPS} Hz     out: {args.out}/\n")

        recording = False
        ep = Episode()
        grip = 0.0
        prev_grip = -1.0   # force one send on first frame
        idx = len([f for f in os.listdir(args.out) if f.endswith(".hdf5")])
        prev = {}
        overruns = 0

        try:
            while True:
                t0 = time.perf_counter()
                btn = js.snapshot_buttons()
                hit = lambda b: btn.get(b, 0) and not prev.get(b, 0)

                if hit(B_QUIT):
                    break

                if hit(B_A):
                    if not recording:
                        ep = Episode(); unwrap.reset(); recording = True
                        print(f"[{idx:03d}] RECORDING")
                    else:
                        recording = False
                        p = os.path.join(args.out, f"episode_{idx:04d}.hdf5")
                        save_episode(ep, p, has_wrist)
                        dd = np.diff(ep.t_mono)
                        print(f"[{idx:03d}] SAVED {len(ep)} frames -> {p}")
                        if len(dd):
                            print(f"      dt mean {dd.mean()*1000:.1f} ms  "
                                  f"max {dd.max()*1000:.1f} ms  target {DT*1000:.1f}")
                        idx += 1

                if hit(B_B) and recording:
                    recording = False
                    print(f"[{idx:03d}] DISCARDED ({len(ep)} frames)")
                    ep = Episode()

                if hit(B_HOME) and not recording:
                    print("  homing...")
                    send_twist(base, (0,0,0), (0,0,0))
                    go_home(base, HOME_JOINTS_DEG)
                    time.sleep(4.0)
                    unwrap.reset()
                    print("  home.")
                    prev = btn
                    continue

                lx, ly = dz(js.axis(AX_LX)), dz(js.axis(AX_LY))
                rx, ry = dz(js.axis(AX_RX)), dz(js.axis(AX_RY))
                lt = (js.axis(AX_LT) + 1) / 2
                rt = (js.axis(AX_RT) + 1) / 2

                vx, vy = -ly*LIN_SPEED, -lx*LIN_SPEED
                vz = (rt - lt) * LIN_SPEED
                wx = (btn.get(B_RB,0) - btn.get(B_LB,0)) * ANG_SPEED
                wy, wz = -ry*ANG_SPEED, -rx*ANG_SPEED

                if btn.get(B_X, 0):
                    grip = min(1.0, grip + GRIPPER_STEP)
                if btn.get(B_Y, 0):
                    grip = max(0.0, grip - GRIPPER_STEP)

                send_twist(base, (vx,vy,vz), (wx,wy,wz))
                if abs(grip - prev_grip) > 1e-6:
                    send_gripper(base, grip)
                    prev_grip = grip

                fb = cyc.RefreshFeedback()
                pos_deg = unwrap([a.position for a in fb.actuators])
                vel = np.array([a.velocity for a in fb.actuators])
                tau = np.array([a.torque for a in fb.actuators])
                b = fb.base
                R = euler_zyx_deg_to_matrix(b.tool_pose_theta_x,
                                            b.tool_pose_theta_y,
                                            b.tool_pose_theta_z)
                g_raw = fb.interconnect.gripper_feedback.motor[0].position
                g = (100.0-g_raw)/100.0 if GRIPPER_INVERT else g_raw/100.0

                fi = front.read()
                wi = wrist.read() if has_wrist else None

                if recording and fi is not None and (not has_wrist or wi is not None):
                    ep.joint_pos_rad.append(np.radians(pos_deg))
                    ep.joint_vel_rad.append(np.radians(vel))
                    ep.joint_torque.append(tau)
                    ep.ee_pos.append([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z])
                    ep.ee_quat.append(matrix_to_quat(R))
                    ep.ee_rotmat.append(R.flatten())
                    ep.ee_euler_deg.append([b.tool_pose_theta_x,
                                            b.tool_pose_theta_y,
                                            b.tool_pose_theta_z])
                    ep.gripper.append(g)
                    ep.cmd_twist.append([vx,vy,vz,wx,wy,wz])
                    ep.cmd_gripper.append(grip)
                    ep.t_mono.append(time.monotonic())
                    ep.img_front.append(fi)
                    if has_wrist:
                        ep.img_wrist.append(wi)

                if fi is not None:
                    disp = fi.copy()
                    tag = "REC" if recording else "idle"
                    col = (0, 0, 255) if recording else (200, 200, 200)
                    cv2.putText(disp, f"{tag}  ep{idx:03d}  n={len(ep)}  grip {g:.2f}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                    cv2.imshow("recorder view", disp)
                    cv2.waitKey(1)

                prev = btn
                el = time.perf_counter() - t0
                if el > DT:
                    overruns += 1
                    if overruns % 20 == 1:
                        print(f"  ! overrun {el*1000:.1f} ms (budget {DT*1000:.1f}) "
                              f"total={overruns}")
                else:
                    time.sleep(DT - el)

        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            send_twist(base, (0,0,0), (0,0,0))
            base.Stop()
            front.release()
            if wrist:
                wrist.release()
            js.close()
            print(f"\ndone. {idx} episodes in {args.out}/  overruns={overruns}")
            os._exit(0)   # joystick thread blocks in read(); force exit


if __name__ == "__main__":
    main()
