#!/usr/bin/env python3
"""Run native Tauri UI acceptance with isolated state on Linux or Windows.

Requires a display and debug desktop/service binaries. All app state, cache,
configuration and API tokens are isolated from the user's running application.
"""
import argparse
from contextlib import contextmanager
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def remove_test_directory(root,windows=None,timeout=15):
    windows=os.name=='nt' if windows is None else windows
    deadline=time.monotonic()+timeout
    while True:
        try:shutil.rmtree(root);return
        except OSError as error:
            if isinstance(error,FileNotFoundError) and not root.exists():return
            # WebView2 may remove/recreate cache entries or retain a handle
            # briefly after the app exits. ERROR_DIR_NOT_EMPTY is an OSError,
            # not a PermissionError, and was reproduced on the Windows runner.
            transient=windows and getattr(error,'winerror',None) in {2,3,5,32,33,145}
            if not transient or time.monotonic()>=deadline:raise
            time.sleep(.2)


@contextmanager
def native_test_directory():
    root=Path(tempfile.mkdtemp(prefix='sayit-native-ui-'))
    try:yield root
    finally:remove_test_directory(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--bin-dir',type=Path,default=Path(__file__).resolve().parents[1]/'target/debug',help='Directory containing debug desktop and service binaries')
    parser.add_argument('--snapshots',type=Path,help='Save native Linux WebKit screenshots in this directory')
    parser.add_argument('--script',type=Path,help='Native UI acceptance script (defaults to player checks)')
    parser.add_argument('--window',choices=['player','main'],default='player')
    parser.add_argument('--voice-file',type=Path,help='Reference audio or exported voice file for management UI acceptance')
    parser.add_argument('--reference-text',help='Exact words spoken in the reference audio for Qwen acceptance')
    parser.add_argument('--timeout',type=float,default=120,help='Native test deadline in seconds')
    parser.add_argument('--model-cache',type=Path,help='Seed isolated model asset directories for a verified-cache install test')
    parser.add_argument('--virtual-display',action='store_true',help='Run on an isolated Linux Xvfb display (requires xvfb-run and Xvfb)')
    parser.add_argument('--pointer-target',type=Path,help='Debug-only geometry output for the isolated native mouse acceptance test')
    args=parser.parse_args()
    if args.virtual_display:
        if not sys.platform.startswith('linux') or not shutil.which('xvfb-run'):
            parser.error('--virtual-display requires Linux, xvfb-run, and Xvfb')
        env=dict(os.environ,GDK_BACKEND='x11');env.pop('WAYLAND_DISPLAY',None)
        command=['xvfb-run','-a',sys.executable,str(Path(__file__).resolve()),*[arg for arg in sys.argv[1:] if arg!='--virtual-display']]
        raise SystemExit(subprocess.run(command,env=env).returncode)
    repo=Path(__file__).resolve().parents[1]
    binaries=args.bin_dir.resolve()
    with native_test_directory() as root:
        fixture={'voice_file':str(args.voice_file.resolve()) if args.voice_file else None,'bad_voice_file':str(root/'invalid.safetensors'),'reference_text':args.reference_text}
        (root/'invalid.safetensors').write_bytes(b'invalid exported voice')
        script=args.script or repo/'scripts/desktop-ui-smoke.js'
        driver=root/'driver.js'
        driver.write_text('window.__SAYIT_UI_FIXTURE__='+json.dumps(fixture)+';\n'+script.read_text())
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env=dict(os.environ,XDG_STATE_HOME=str(root/'state'),XDG_CONFIG_HOME=str(root/'config'),XDG_CACHE_HOME=str(root/'cache'),
                 SAYIT_DESKTOP_TEST_MODE='1',SAYIT_SERVICE_URL=f'http://127.0.0.1:{port}/v1',
                 SAYIT_UI_TEST_SCRIPT=str(driver),SAYIT_UI_TEST_RESULT=str(root/'result.json'),SAYIT_UI_TEST_WINDOW=args.window)
        if os.name=='nt':
            # Microsoft documents this override for isolated WebView2 profiles:
            # https://learn.microsoft.com/microsoft-edge/webview2/reference/win32/webview2-idl
            env.update(LOCALAPPDATA=str(root/'state'),APPDATA=str(root/'config'),WEBVIEW2_USER_DATA_FOLDER=str(root/'cache/webview'))
        suffix='.exe' if os.name=='nt' else ''
        data=root/'state/sayit'
        if args.model_cache:
            shutil.copytree(args.model_cache,data/'models',dirs_exist_ok=True)
        if args.snapshots:
            args.snapshots.mkdir(parents=True,exist_ok=True)
            env['SAYIT_UI_TEST_SNAPSHOTS']=str(args.snapshots.resolve())
        if args.pointer_target:
            env['SAYIT_UI_TEST_POINTER_TARGET']=str(args.pointer_target.resolve())
        with (root/'service.log').open('w+') as service_log,(root/'desktop.log').open('w+') as desktop_log:
            service=subprocess.Popen([str(binaries/('sayit-service'+suffix)),'--port',str(port),'--data-dir',str(data),'--config',str(args.config.resolve())],
                                     env=env,stdout=service_log,stderr=subprocess.STDOUT)
            desktop=None
            try:
                deadline=time.monotonic()+15
                while True:
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/health',timeout=1):break
                    except OSError:
                        assert service.poll() is None,'service exited during startup'
                        assert time.monotonic()<deadline,'service startup timed out'
                        time.sleep(0.1)
                desktop=subprocess.Popen([str(binaries/('sayit-desktop'+suffix))],env=env,stdout=desktop_log,stderr=subprocess.STDOUT)
                desktop.wait(timeout=args.timeout)
                assert (root/'result.json').exists(),'native UI did not return an acceptance result'
                result=json.loads((root/'result.json').read_text())
                print(json.dumps(result,indent=2))
                assert result.get('passed'),result.get('error')
                assert desktop.returncode==0,'desktop exited with a failure'
            except Exception:
                for label,log in [('desktop',desktop_log),('service',service_log)]:
                    log.flush();log.seek(0);print(label+' log:\n'+log.read()[-6000:])
                raise
            finally:
                if service.poll() is None:
                    try:
                        request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/playback/stop',data=b'{}',
                            headers={'Authorization':'Bearer '+(data/'api-token').read_text().strip(),'Content-Type':'application/json'})
                        with urllib.request.urlopen(request,timeout=2):pass
                    except (OSError,ValueError):pass
                for process in [desktop,service]:
                    if process is not None and process.poll() is None:
                        process.terminate()
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
