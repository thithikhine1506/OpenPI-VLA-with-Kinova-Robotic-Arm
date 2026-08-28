#!/usr/bin/env python3
"""
Kinova Gen3 policy rollout client.

Reads RealSense + Kortex state, queries the openpi policy server, and drives
the arm with the returned joint position targets.

Run in the kinova_env venv on the lab machine:
    source ~/kinova_env/bin/activate
    python3 gen3_policy_client.py --host https://xxxx.trycloudflare.com

SAFETY -- READ BEFORE RUNNING
    * DEADMAN SWITCH: the arm only moves while you HOLD the deadman button.
      Release it and the arm stops immediately. Default is RB (button 5).
    * Keep a hand on the physical e-stop for the first rollouts.
    * MAX_JOINT_SPEED and MAX_TARGET_JUMP clamp anything the policy asks for.
      They start deliberately conservative. Raise them only once you have
      watched several rollouts behave sensibly.
    * Press the QUIT button to exit cleanly.

THE WRAP -- the single most important line in this file
    Training used joint angles canonicalized into [-pi, pi] per episode
    (see canonicalize_branch in convert_kinova_to_lerobot.py). j0 and j4 sit on
    the 0/2*pi boundary at the home pose, so raw Kortex readings land on either
    branch. Send un-wrapped angles and the model sees a pose it never trained
    on -- silently, with no error. wrap_to_pi() below is what prevents that.
"""

import argparse
import collections
import json
import math
import os
import struct
import sys
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from openpi_client import websocket_client_policy

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

# Must match the recording rate. Deploying at a different rate than you trained
# at changes the system dynamics and degrades performance even when the policy
# is fine.
FPS = 15
DT = 1.0 / FPS

HOME_JOINTS_DEG = [0.280, 3.684, 184.813, 240.679, 359.782, 304.277, 95.178]

IMG_W, IMG_H = 640, 480
IMG_OUT = 256          # must match the converter's output size

JS_DEV = "/dev/input/js0"
PADMAP = "padmap.json"

# --- Safety limits. Start here; loosen only after watching real rollouts. ---
MAX_JOINT_SPEED = 20.0     # deg/s, hard clamp on commanded joint velocity
MAX_TARGET_JUMP = 0.35     # rad, reject a target further than this from current
KP = 2.0                   # proportional gain, target error -> joint speed

# How many of the 10 returned actions to execute before re-querying.
# All 10 = 0.67 s open loop, fewer queries, more latency tolerance.
# Fewer  = more reactive, more queries, more sensitive to tunnel latency.
ACTIONS_PER_QUERY = 3

GRIPPER_EPS = 0.02         # only send a gripper command when it moves this much

# --- Table-collision recovery ---
# If the policy commands motion but the joints stop moving, the arm is pressing
# into something. Lift straight up in the base frame and let the policy re-plan
# from a clear pose rather than grinding against the table.
STALL_WINDOW = 12          # frames (~0.8 s at 15 Hz) of history to check
STALL_MOVE = 0.006         # rad; less total movement than this = stalled
STALL_CMD = 0.02           # rad; only counts as a stall if we ASKED for motion
LIFT_SPEED = 0.06          # m/s upward during recovery
LIFT_TIME = 0.7            # s

# --- Table floor ---
# Across 54 demos the end effector never went below z = 0.0058 m (1st pct
# 0.0260, median 0.1445). Below the floor the arm is pressing into the table,
# not reaching for anything. Clamp downward motion there.
Z_FLOOR = 0.010            # m in the base frame
Z_ESCAPE = 0.05            # m/s upward when below the floor


# Mean joint positions from the training norm stats (state dims 0..6).
TRAIN_JOINT_MEAN = np.array([-0.32640389, 0.46963099, -3.14629030,
                             -1.60254872, 0.02341108, -1.06626344,
                             1.23523057])


def wrap_to_pi(rad):
    """Plain wrap into [-pi, pi]. For error deltas, NOT for observations."""
    return (np.asarray(rad) + np.pi) % (2 * np.pi) - np.pi


def canonicalize_state(rad):
    """Put each joint on the 2*pi branch nearest the TRAINING distribution.

    Wrapping to [-pi, pi] is not enough: j2 sits on the +/-pi boundary, so raw
    readings land at +3.142 while training data has j2 around -3.146 -- same
    pose, 2*pi apart. Against q01=-3.48/q99=-2.45 that normalizes to about
    +11.8 instead of something inside [-1, 1].
    """
    return TRAIN_JOINT_MEAN + wrap_to_pi(np.asarray(rad) - TRAIN_JOINT_MEAN)


# ----------------------------------------------------------------------------
# Joystick (deadman + quit only)
# ----------------------------------------------------------------------------

class Joystick:
    FMT, SZ = "IhBB", 8

    def __init__(self, dev=JS_DEV):
        self.f = open(dev, "rb")
        self.buttons = {}
        self._lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()
        time.sleep(0.2)

    def _loop(self):
        while not self._stop:
            ev = self.f.read(self.SZ)
            if not ev:
                break
            _, val, typ, num = struct.unpack(self.FMT, ev)
            if (typ & ~0x80) == 0x01:
                with self._lock:
                    self.buttons[num] = val

    def button(self, n):
        with self._lock:
            return self.buttons.get(n, 0)

    def close(self):
        self._stop = True
        try:
            self.f.close()
        except Exception:
            pass


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


def send_joint_speeds(base, speeds_deg_s):
    js = Base_pb2.JointSpeeds()
    for i, s in enumerate(speeds_deg_s):
        j = js.joint_speeds.add()
        j.joint_identifier = i
        j.value = float(np.clip(s, -MAX_JOINT_SPEED, MAX_JOINT_SPEED))
        j.duration = 0
    base.SendJointSpeedsCommand(js)


def stop_arm(base):
    send_joint_speeds(base, [0.0] * 7)


def send_gripper(base, value):
    cmd = Base_pb2.GripperCommand()
    cmd.mode = Base_pb2.GRIPPER_POSITION
    f = cmd.gripper.finger.add()
    f.finger_identifier = 1
    f.value = float(np.clip(value, 0.0, 1.0))
    base.SendGripperCommand(cmd)


def send_twist_z(base, vz):
    """Cartesian velocity in the BASE frame -- used only for collision recovery."""
    cmd = Base_pb2.TwistCommand()
    cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    cmd.duration = 0
    cmd.twist.linear_x = 0.0
    cmd.twist.linear_y = 0.0
    cmd.twist.linear_z = float(vz)
    cmd.twist.angular_x = 0.0
    cmd.twist.angular_y = 0.0
    cmd.twist.angular_z = 0.0
    base.SendTwistCommand(cmd)


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
# Camera
# ----------------------------------------------------------------------------

class Camera:
    def __init__(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
        self.pipe.start(cfg)

    def read(self):
        try:
            f = self.pipe.wait_for_frames(timeout_ms=1000).get_color_frame()
            if not f:
                return None
            # .copy() -- asanyarray returns a view into librealsense's buffer;
            # holding views exhausts the 16-frame pool.
            return np.asanyarray(f.get_data()).copy()
        except Exception:
            return None

    def release(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


def to_model_image(bgr):
    """640x480 BGR -> 256x256 RGB. Must match the converter exactly."""
    img = cv2.resize(bgr, (IMG_OUT, IMG_OUT), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True,
                    help="Policy server URL, e.g. https://xxx.trycloudflare.com "
                         "or 'localhost' for a local server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ip", default=ROBOT_IP)
    ap.add_argument("--prompt", default=TASK)
    ap.add_argument("--dry-run", action="store_true",
                    help="Query the policy and print actions, but never move the arm")
    args = ap.parse_args()

    if not os.path.exists(PADMAP):
        sys.exit(f"No {PADMAP} -- run gen3_recorder.py --calibrate first.")
    pad = json.load(open(PADMAP))
    B_DEADMAN = pad["RB"]
    B_QUIT = pad["QUIT"]
    B_HOME = pad["HOME"]

    print("opening joystick...", flush=True)
    js = Joystick()
    print("joystick ok. opening camera...", flush=True)
    cam = Camera()
    print("camera ok. connecting to policy server...", flush=True)

    # WebsocketClientPolicy passes through any host starting with "ws".
    # Cloudflare terminates TLS, so a tunnelled server needs wss:// on 443 --
    # plain ws:// to 443 gets an HTTP 400 at the handshake.
    raw = args.host.rstrip("/")
    if raw.startswith("https://"):
        uri, port = "wss://" + raw[len("https://"):], None
    elif raw.startswith("http://"):
        uri, port = "ws://" + raw[len("http://"):], None
    elif raw.startswith("ws://") or raw.startswith("wss://"):
        uri, port = raw, None
    else:
        uri, port = raw, args.port

    print(f"connecting to policy server {uri}{'' if port is None else ':' + str(port)} ...")
    policy = websocket_client_policy.WebsocketClientPolicy(host=uri, port=port)
    print("connected. metadata:", policy.get_server_metadata())

    print(f"\ntask   : {args.prompt}")
    print(f"rate   : {FPS} Hz    actions/query: {ACTIONS_PER_QUERY}")
    print(f"limits : {MAX_JOINT_SPEED} deg/s, {MAX_TARGET_JUMP} rad jump")
    print(f"\nHOLD button {B_DEADMAN} (RB) to enable motion. Release = STOP.")
    print(f"Button {B_HOME} = go home.   Button {B_QUIT} = quit.\n")

    with DeviceConnection(args.ip) as router:
        base = BaseClient(router)
        cyc = BaseCyclicClient(router)
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(mode)

        prev_grip = -1.0
        cam_fail = 0
        queries = 0
        recoveries = 0
        pos_hist = collections.deque(maxlen=STALL_WINDOW)
        cmd_hist = collections.deque(maxlen=STALL_WINDOW)
        rejected = 0

        print("entering control loop", flush=True)
        try:
            while True:
                if js.button(B_QUIT):
                    print("QUIT pressed", flush=True)
                    break

                if js.button(B_HOME) and not js.button(B_DEADMAN):
                    print("  homing...")
                    stop_arm(base)
                    go_home(base, HOME_JOINTS_DEG)
                    time.sleep(4.0)
                    print("  home.")
                    continue

                # ---- observe ----
                fb = cyc.RefreshFeedback()
                jp_rad = np.radians([a.position for a in fb.actuators])
                jp_wrapped = canonicalize_state(jp_rad)  # <<< THE WRAP
                g_raw = fb.interconnect.gripper_feedback.motor[0].position
                grip = g_raw / 100.0

                bgr = cam.read()
                if bgr is None:
                    cam_fail += 1
                    if cam_fail % 20 == 1:
                        print(f"  ! camera returned None ({cam_fail} times)")
                    continue

                state = np.concatenate([jp_wrapped, [grip]]).astype(np.float32)

                obs = {
                    "observation/image": to_model_image(bgr),
                    "observation/state": state,
                    "prompt": args.prompt,
                }

                # ---- query ----
                t0 = time.perf_counter()
                result = policy.infer(obs)
                latency = (time.perf_counter() - t0) * 1000
                actions = np.asarray(result["actions"])   # (10, 8)
                queries += 1

                if queries % 5 == 1:
                    dm = js.button(B_DEADMAN)
                    print(f"[q{queries}] deadman={'HELD' if dm else 'released'}  "
                          f"lat {latency:.0f}ms  "
                          f"j={np.round(jp_wrapped, 2)}  grip {grip:.2f}")

                if queries == 1:
                    print(f"first action chunk: shape {actions.shape}, "
                          f"latency {latency:.0f} ms")
                    print(f"  state sent    : {np.round(state, 3)}")
                    print(f"  action[0]     : {np.round(actions[0], 3)}")
                    if actions.shape[1] != 8:
                        stop_arm(base)
                        sys.exit(f"Expected 8 action dims, got {actions.shape[1]}")

                # ---- execute ----
                for k in range(min(ACTIONS_PER_QUERY, len(actions))):
                    step_t0 = time.perf_counter()

                    if js.button(B_QUIT):
                        break
                    if not js.button(B_DEADMAN):
                        stop_arm(base)
                        time.sleep(DT)
                        continue

                    fb = cyc.RefreshFeedback()
                    cur = canonicalize_state(np.radians([a.position for a in fb.actuators]))

                    target = actions[k][:7]
                    err = wrap_to_pi(target - cur)       # shortest path, no 2pi jumps

                    if np.abs(err).max() > MAX_TARGET_JUMP:
                        rejected += 1
                        stop_arm(base)
                        if rejected % 10 == 1:
                            print(f"  ! target rejected: max err "
                                  f"{np.abs(err).max():.3f} rad > {MAX_TARGET_JUMP} "
                                  f"(total {rejected})")
                        break

                    # --- table floor guard ---
                    z_now = fb.base.tool_pose_z
                    if z_now < Z_FLOOR and not args.dry_run:
                        recoveries += 1
                        if recoveries % 5 == 1:
                            print(f"  ! below table floor (z={z_now:.4f} < {Z_FLOOR}) "
                                  f"-- lifting [{recoveries}]")
                        stop_arm(base)
                        send_twist_z(base, Z_ESCAPE)
                        time.sleep(0.15)
                        send_twist_z(base, 0.0)
                        stop_arm(base)
                        pos_hist.clear(); cmd_hist.clear()
                        break

                    pos_hist.append(cur.copy())
                    cmd_hist.append(np.abs(err).max())

                    stalled = (
                        len(pos_hist) == STALL_WINDOW
                        and np.abs(np.array(pos_hist) - pos_hist[0]).max() < STALL_MOVE
                        and min(cmd_hist) > STALL_CMD
                    )

                    if stalled and not args.dry_run:
                        recoveries += 1
                        print(f"  ! stalled (likely table contact) -- lifting "
                              f"[recovery {recoveries}]")
                        stop_arm(base)
                        t_lift = time.perf_counter()
                        while time.perf_counter() - t_lift < LIFT_TIME:
                            if not js.button(B_DEADMAN):
                                break
                            send_twist_z(base, LIFT_SPEED)
                            time.sleep(0.05)
                        send_twist_z(base, 0.0)
                        stop_arm(base)
                        pos_hist.clear()
                        cmd_hist.clear()
                        break          # re-query from the lifted pose

                    if args.dry_run:
                        stop_arm(base)
                    else:
                        send_joint_speeds(base, np.degrees(KP * err))
                        g_cmd = float(actions[k][7])
                        if abs(g_cmd - prev_grip) > GRIPPER_EPS:
                            send_gripper(base, g_cmd)
                            prev_grip = g_cmd

                    el = time.perf_counter() - step_t0
                    if el < DT:
                        time.sleep(DT - el)

        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            # Wrap teardown: if the interrupt lands mid-RPC these can hang.
            for fn in (lambda: stop_arm(base), base.Stop):
                try:
                    fn()
                except Exception:
                    pass
            cam.release()
            js.close()
            print(f"\ndone. {queries} queries, {rejected} rejected, {recoveries} recoveries")
            os._exit(0)


if __name__ == "__main__":
    main()
