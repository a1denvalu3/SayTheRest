#!/usr/bin/env python3
"""Run native Tauri UI acceptance with a real preset model and silent audio.

Requires a display and debug desktop/service binaries. All app state, cache,
configuration and API tokens are isolated from the user's running application.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--snapshots',type=Path,help='Save native Linux WebKit screenshots in this directory')
    parser.add_argument('--script',type=Path,help='Native UI acceptance script (defaults to player checks)')
    parser.add_argument('--window',choices=['player','main'],default='player')
    parser.add_argument('--voice-file',type=Path,help='Reference audio or exported voice file for management UI acceptance')
    parser.add_argument('--timeout',type=float,default=120,help='Native test deadline in seconds')
    parser.add_argument('--model-cache',type=Path,help='Seed isolated model asset directories for a verified-cache install test')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='sayit-native-ui-') as temporary:
        root=Path(temporary)
        fixture={'voice_file':str(args.voice_file.resolve()) if args.voice_file else None,'bad_voice_file':str(root/'invalid.safetensors')}
        (root/'invalid.safetensors').write_bytes(b'invalid exported voice')
        script=args.script or repo/'scripts/desktop-ui-smoke.js'
        driver=root/'driver.js'
        driver.write_text('window.__SAYIT_UI_FIXTURE__='+json.dumps(fixture)+';\n'+script.read_text())
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env=dict(os.environ,XDG_STATE_HOME=str(root/'state'),XDG_CONFIG_HOME=str(root/'config'),XDG_CACHE_HOME=str(root/'cache'),
                 SAYIT_DESKTOP_TEST_MODE='1',SAYIT_SERVICE_URL=f'http://127.0.0.1:{port}/v1',
                 SAYIT_UI_TEST_SCRIPT=str(driver),SAYIT_UI_TEST_RESULT=str(root/'result.json'),SAYIT_UI_TEST_WINDOW=args.window)
        data=root/'state/sayit'
        if args.model_cache:
            shutil.copytree(args.model_cache,data/'models',dirs_exist_ok=True)
        if args.snapshots:
            args.snapshots.mkdir(parents=True,exist_ok=True)
            env['SAYIT_UI_TEST_SNAPSHOTS']=str(args.snapshots.resolve())
        with (root/'service.log').open('w+') as service_log,(root/'desktop.log').open('w+') as desktop_log:
            service=subprocess.Popen([str(repo/'target/debug/sayit-service'),'--port',str(port),'--data-dir',str(data),'--config',str(args.config.resolve())],
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
                desktop=subprocess.Popen([str(repo/'target/debug/sayit-desktop')],env=env,stdout=desktop_log,stderr=subprocess.STDOUT)
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
                for process in [desktop,service]:
                    if process is not None and process.poll() is None:
                        process.terminate()
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
