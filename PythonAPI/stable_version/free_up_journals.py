#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:54:14 2021

@author: altius
"""
import os
import time

import time
while True:
    path='/var/log/journal/916fc3f8d8f54e64a042f83ce0f09dba/'
    os.chdir(path)
    os.system('sudo journalctl --vacuum-size=10M')
    time.sleep(20)
