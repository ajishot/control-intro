import cv2
from pymavlink import mavutil
#from dt_apriltags import Detector
import matplotlib.pyplot as plt
import numpy as np
from pid import PID
import sys
import signal
import apriltaggy

def set_rc_channel_pwm(mav, channel_id, pwm=1500):
    """Set RC channel pwm value
    Args:a
        channel_id (TYPE): Channel ID
        pwm (int, optional): Channel pwm value 1100-1900
    """
    if channel_id < 1 or channel_id > 18:
        print("Channel does not exist.")
        return

    # Mavlink 2 supports up to 18 channels:
    # https://mavlink.io/en/messages/common.html#RC_CHANNELS_OVERRIDE
    rc_channel_values = [65535 for _ in range(18)]
    rc_channel_values[channel_id - 1] = pwm
    mav.mav.rc_channels_override_send(
        mav.target_system,  # target_system
        mav.target_component,  # target_component
        *rc_channel_values,
    )

def set_vertical_power(mav, power=0):
    """Set vertical power
    Args:
        power (int, optional): Vertical power value -100-100
    """
    if power < -100 or power > 100:
        print("Power value out of range. Clipping...")
        power = np.clip(power, -100, 100)

    power = int(power)

    set_rc_channel_pwm(mav, 3, 1500 + power * 5)

def set_lateral_power(mav, power=0):
    """Set vertical power
    Args:
        power (int, optional): Vertical power value -100-100
    """
    if power < -100 or power > 100:
        print("Power value out of range. Clipping...")
        power = np.clip(power, -100, 100)

    power = int(power)

    set_rc_channel_pwm(mav, 6, 1500 + power * 5)