#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2022, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
# Notice
#   1. Changes to this file on Studio will not be preserved
#   2. The next conversion will overwrite the file with the same name
# 
# xArm-Python-SDK: https://github.com/xArm-Developer/xArm-Python-SDK
#   1. git clone git@github.com:xArm-Developer/xArm-Python-SDK.git
#   2. cd xArm-Python-SDK
#   3. python setup.py install
"""
import sys
import math
import time
import queue
import datetime
import random
import traceback
import threading
from xarm import version
from xarm.wrapper import XArmAPI

import sys

def wait_for_space(prompt="Press SPACE to continue… (q/ESC to abort)"):
    print(prompt, flush=True)
    # macOS/Linux
    import termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == ' ':
                return
            if ch in ('\x1b', 'q', 'Q', '\x03', '\x04'):  # ESC/q/Ctrl-C/Ctrl-D
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class RobotMain(object):
    """Robot Main Class"""
    def __init__(self, robot, **kwargs):
        self.alive = True
        self._arm = robot
        self._ignore_exit_state = False
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500
        self._vars = {}
        self._funcs = {}
        self._robot_init()

    # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if not self._ignore_exit_state and data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code, self._arm.connected, self._arm.state, self._arm.error_code, ret1, ret2))
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1], ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    @property
    def arm(self):
        return self._arm

    @property
    def VARS(self):
        return self._vars

    @property
    def FUNCS(self):
        return self._funcs

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._ignore_exit_state:
                return True
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    # Robot Main Run
    def in_run(self):
        try:
            code = self._arm.set_state(0)
            if not self._check_code(code, 'set_state'):
                return
            
            code = self._arm.set_servo_angle(angle=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            for i in range(100):
                time.sleep(0.1)
                if not self.is_alive:
                    return
            if not self._check_code(code, 'set_servo_angle'):

                return
            
            code = self._arm.set_servo_angle(angle=[0, 1.7, -7, 62, 0, 41.6, -93], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
            
            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):

                return
            
            code = self._arm.set_servo_angle(angle=[0, 19, -3.4, 91.5, 0, 66.4, -93], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
            
            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):

                return
            
            code = self._arm.set_servo_angle(angle=[0, 15, -4, 5.7, 0, -69, -93], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
            
            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return         

            code = self._arm.set_servo_angle(angle=[-21.7, 49.8, -26, 50.5, -70, -87, -80], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return         

            code = self._arm.set_servo_angle(angle=[-21.8, 28.9, -16, 45.3, -71.2, -57.9, -68.5], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[-21.5, 19.7, -16.6, 63.7, -105.7, -47, -39], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
            
            wait_for_space()
            
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[-49, 86.5, -2.7, 133.6, -104.6, -95, -45.5], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()


            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[-35.5, 62.7, -27.1, 136, -106.8, -95, -24], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
            

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[-18.5, 46.3, -22, 131, -130.4, -82.3, -7.7], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[-39.4, 66, 15, 138.1, -140.7, -75.6, -15.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
            
            wait_for_space()    

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[-10.8, 34.1, -12.7, 118.3, -150.2, -66.8, 3.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[37.9, 37.4, -48.3, 112.7, -158.7, -65.8, 128.4], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[62, 56.3, -65.4, 106.8, -165, -59, 158], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[89.7, 99.6, -77.4, 108, -165, -65.5, 190.3], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[64.3, 99.5, -80, 134.6, -165, -72.6, 186.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[53, 71.7, -67.5, 136.3, -163.3, -77.3, 178.7], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()
            
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[40, 57.8, -52.8, 136, -163.6, -77.8, 172], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[18, 48.5, -25.7, 133.9, -163, -71, 89.7], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[0, 0, 0, 0, 0, 0, 0], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            return
            if not self._check_code(code, 'set_servo_angle'):
                return
        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        finally:
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)

    def out_run(self):
        try:
            code = self._arm.set_state(0)
            if not self._check_code(code, 'set_state'):
                return
            
            code = self._arm.set_servo_angle(angle=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
            
            if not self._check_code(code, 'set_servo_angle'):
                return
            for i in range(100):
                time.sleep(0.1)
                if not self.is_alive:
                    return
            
            if not self._check_code(code, 'set_servo_angle'):
                return

            code = self._arm.set_servo_angle(angle=[-2.9, 3.9, 5.5, 68.7, -9.1, 56.6, -81.5], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
            
            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[-3, -20.7, 2, 36.6, -9.1, 33.3, -81.5], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
            
            wait_for_space()
            
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[-2.9, 18.8, -9, 1.1, -13.8, -78.3, -90.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[48.6, 13.3, -31.7, 86.7, 166.8, -56.7, -188], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[77.8, 62.7, -55, 100.8, 170.6, -65.3, 171.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
            
            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[92.4, 89.5, -60.3, 107.1, 176.2, -72.2, -157.5], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[82.9, 10.5, -72.2, 64.3, 173.5, -34.4, -221.3], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[112.7, 82.3, -87.6, 65.6, 207.5, -48.7, -204.1], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[91.9, -21.8, -107.4, 64.4, 192.1, -25.9, -325.5], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[68.4, -87.8, -102.4, 71.9, 163, -40.5, -357.5], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[107, -95.1, -98.5, 129.1, 162.7, -74.1, -358.1], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[103.3, -65.5, -102, 112.4, 162.7, -70.5, -358.1], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[103.8, -47.9, -102, 101.4, 162.7, -61.3, -358.1], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[108.7, -30.2, -105.3, 92.1, 161.8, -57.7, -318.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[132.5, -45.2, -109.5, 105.9, 142.4, -70.7, -272.9], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[145.9, -44, -108.9, 112.7, 132.5, -86.4, -237.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()

            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[163, -42.2, -125, 109.6, 132.5, -83, -193,7], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            wait_for_space()
            
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[0,0,0,0,0,0,0], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            if not self._check_code(code, 'set_servo_angle'):
                return
        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        finally:
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)


if __name__ == '__main__':
    RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm_in = XArmAPI('192.168.1.248', baud_checkset=False)
    time.sleep(0.5)
    arm_out = XArmAPI('192.168.1.200', baud_checkset=False)

    # robot_in = RobotMain(arm_in)
    # robot_in.in_run()

    robot_out = RobotMain(arm_out)
    robot_out.out_run()