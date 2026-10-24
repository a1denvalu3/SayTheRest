#!/usr/bin/env python3
"""Real mouse acceptance for the Windows WebView2 player menu and focus dismissal.

Requires an interactive Windows desktop and debug binaries. The fixture contains
UI metadata only; no inference or audio-device support is established here.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir',type=Path,default=Path(__file__).resolve().parents[1]/'target/debug')
    args=parser.parse_args()
    if os.name!='nt':parser.error('An interactive Windows desktop is required')
    user=ctypes.WinDLL('user32',use_last_error=True)
    def api(name,arguments,result):
        function=getattr(user,name);function.argtypes=arguments;function.restype=result;return function
    set_dpi=api('SetProcessDpiAwarenessContext',[ctypes.c_void_p],wintypes.BOOL);set_dpi(ctypes.c_void_p(-4))
    create=api('CreateWindowExW',[wintypes.DWORD,wintypes.LPCWSTR,wintypes.LPCWSTR,wintypes.DWORD,
        ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,wintypes.HWND,wintypes.HMENU,wintypes.HINSTANCE,ctypes.c_void_p],wintypes.HWND)
    destroy=api('DestroyWindow',[wintypes.HWND],wintypes.BOOL)
    find=api('FindWindowW',[wintypes.LPCWSTR,wintypes.LPCWSTR],wintypes.HWND)
    shown=api('IsWindowVisible',[wintypes.HWND],wintypes.BOOL)
    rect=api('GetWindowRect',[wintypes.HWND,ctypes.POINTER(wintypes.RECT)],wintypes.BOOL)
    client_rect=api('GetClientRect',[wintypes.HWND,ctypes.POINTER(wintypes.RECT)],wintypes.BOOL)
    to_screen=api('ClientToScreen',[wintypes.HWND,ctypes.POINTER(wintypes.POINT)],wintypes.BOOL)
    move=api('MoveWindow',[wintypes.HWND,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,wintypes.BOOL],wintypes.BOOL)
    metrics=api('GetSystemMetrics',[ctypes.c_int],ctypes.c_int)
    cursor=api('SetCursorPos',[ctypes.c_int,ctypes.c_int],wintypes.BOOL)
    mouse=api('mouse_event',[wintypes.DWORD,wintypes.DWORD,wintypes.DWORD,wintypes.DWORD,ctypes.c_size_t],None)
    callback=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    enumerate_windows=api('EnumWindows',[callback,wintypes.LPARAM],wintypes.BOOL)
    peek=api('PeekMessageW',[ctypes.POINTER(wintypes.MSG),wintypes.HWND,wintypes.UINT,wintypes.UINT,wintypes.UINT],wintypes.BOOL)
    translate=api('TranslateMessage',[ctypes.POINTER(wintypes.MSG)],wintypes.BOOL)
    dispatch=api('DispatchMessageW',[ctypes.POINTER(wintypes.MSG)],ctypes.c_ssize_t)
    def pump():
        message=wintypes.MSG()
        while peek(ctypes.byref(message),None,0,0,1):translate(ctypes.byref(message));dispatch(ctypes.byref(message))
    def visible():
        result=set()
        @callback
        def collect(window,_):
            if shown(window):result.add(window)
            return True
        enumerate_windows(collect,0);return result
    def click(x,y):
        assert cursor(x,y),'could not move the desktop pointer'
        mouse(2,0,0,0,0);mouse(4,0,0,0,0);pump()
    def click_client(window,x,y):
        point=wintypes.POINT(round(x),round(y));assert to_screen(window,ctypes.byref(point));click(point.x,point.y)
    repo=Path(__file__).resolve().parents[1]
    driver=runpy.run_path(str(repo/'scripts/test-player-native-menu.py'))['SCRIPT']
    outside=create(0,'STATIC','SayIt acceptance outside target',0x10CF0000,0,0,180,90,None,None,None,None)
    assert outside,'could not create the isolated outside-click target'
    try:
        with tempfile.TemporaryDirectory(prefix='sayit-windows-menu-') as temporary:
            root=Path(temporary);script=root/'driver.js';script.write_text(driver);target=root/'target.json'
            model=root/'ui-only.onnx';tokens=root/'tokens.txt';model.write_bytes(b'UI fixture; inference is not called');tokens.write_text('UI fixture')
            config=root/'config.json';config.write_text(json.dumps({'engine':'sherpa-onnx-vits','executable':'unused-ui-fixture','model':str(model),'tokens':str(tokens),'provider':'cpu'}))
            with (root/'runner.log').open('w+') as log:
                runner=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/test-desktop-ui.py'),'--config',str(config),'--bin-dir',str(args.bin_dir.resolve()),'--script',str(script),'--pointer-target',str(target),'--timeout','60'],stdout=log,stderr=subprocess.STDOUT)
                try:
                    deadline=time.monotonic()+80
                    while not target.exists():
                        pump();assert runner.poll() is None,'native driver exited before readiness';assert time.monotonic()<deadline,'native menu readiness timed out';time.sleep(.05)
                    geometry=json.loads(target.read_text());window=find(None,'sayIt player')
                    assert window and shown(window),'native player window not found'
                    bounds=wintypes.RECT();assert client_rect(window,ctypes.byref(bounds))
                    assert bounds.right==round(geometry['width']*geometry['scale']) and bounds.bottom==round(geometry['height']*geometry['scale']),'DPI/client geometry does not match the actual webview'
                    assert rect(window,ctypes.byref(bounds))
                    outside_x=0 if bounds.left>=190 else metrics(0)-190
                    assert move(outside,outside_x,0,180,90,True)
                    before=visible();click_client(window,geometry['x']*geometry['scale'],geometry['y']*geometry['scale'])
                    time.sleep(.75);pump();opened=visible()
                    assert window in opened,'player dismissed when its native menu opened'
                    assert opened-before,'no native popup appeared after real mouse input'
                    click_client(window,200*geometry['scale'],20*geometry['scale']);time.sleep(.5);pump()
                    assert shown(window),'player dismissed while closing its own menu'
                    time.sleep(4);click(outside_x+50,40)
                    deadline=time.monotonic()+25
                    while runner.poll() is None:
                        pump();assert time.monotonic()<deadline,'Windows native menu acceptance timed out';time.sleep(.05)
                    assert runner.returncode==0,'Windows native menu acceptance failed'
                    log.flush();log.seek(0);print(log.read())
                except Exception:
                    log.flush();log.seek(0);print(log.read());raise
                finally:
                    if runner.poll() is None:
                        runner.terminate()
                        try:runner.wait(timeout=15)
                        except subprocess.TimeoutExpired:runner.kill();runner.wait()
    finally:destroy(outside)


if __name__=='__main__':main()
