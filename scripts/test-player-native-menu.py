#!/usr/bin/env python3
"""Exercise the actual GTK select menu with mouse input on a private Xvfb display.

This is a focus/menu test, with no inference. --config can use installed-model
metadata; otherwise a small UI-only file fixture bootstraps the model picker.
Requires debug desktop/service binaries, xvfb-run, Xvfb, xauth, and xdotool.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

SCRIPT=r"""(async()=>{
const invoke=window.__TAURI__.core.invoke,checks=[];
const until=async run=>{const end=performance.now()+15000;while(performance.now()<end){const result=await run();if(result)return result;await new Promise(resolve=>setTimeout(resolve,150));}throw Error('Native menu acceptance timed out')};
try{
 await invoke('open_player');
 const select=await until(()=>{const element=document.querySelector('#player-model');return element?.value?element:null});
 const rect=select.getBoundingClientRect();
 await window.__TAURI__.event.emit('ui-test-pointer-target',{x:(rect.left+rect.right)/2,y:(rect.top+rect.bottom)/2,width:innerWidth,height:innerHeight,scale:devicePixelRatio});
 const end=performance.now()+3000;
 while(performance.now()<end){if(!await invoke('window_visible'))throw Error('Player hid while its native select menu was open');await new Promise(resolve=>setTimeout(resolve,150));}
 checks.push('native select menu opens and closes without dismissing the player');
 await until(async()=>!await invoke('window_visible'));
 checks.push('clicking outside the app still dismisses the player');
 await window.__TAURI__.event.emit('ui-test-result',{passed:true,checks});
}catch(error){let models;try{models=(await invoke('service_get',{resource:'models'})).map(model=>({id:model.id,installed:model.installed,selected:model.selected}))}catch{}await window.__TAURI__.event.emit('ui-test-result',{passed:false,error:String(error),checks,windowVisible:await invoke('window_visible'),viewVisible:typeof visible==='boolean'?visible:null,choices:document.querySelector('#player-model')?.innerHTML,models})}
})();"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--bin-dir',type=Path,default=Path(__file__).resolve().parents[1]/'target/debug')
    parser.add_argument('--display-child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not sys.platform.startswith('linux') or not all(shutil.which(tool) for tool in ['xvfb-run','Xvfb','xdotool']):
        parser.error('Linux, xvfb-run, Xvfb, and xdotool are required')
    if not args.display_child:
        env=dict(os.environ,GDK_BACKEND='x11');env.pop('WAYLAND_DISPLAY',None)
        raise SystemExit(subprocess.run(['xvfb-run','-a',sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--display-child'],env=env).returncode)
    repo=Path(__file__).resolve().parents[1]
    def xdo(*arguments):return subprocess.run(['xdotool',*map(str,arguments)],capture_output=True,text=True)
    def visible():return set(xdo('search','--onlyvisible','--name','.*').stdout.splitlines())
    with tempfile.TemporaryDirectory(prefix='sayit-native-menu-') as temporary:
        root=Path(temporary);script=root/'driver.js';script.write_text(SCRIPT);target=root/'target.json'
        config=args.config
        if config is None:
            model=root/'ui-only.onnx';tokens=root/'tokens.txt';model.write_bytes(b'UI metadata fixture; inference is not called');tokens.write_text('UI fixture')
            config=root/'config.json';config.write_text(json.dumps({'engine':'sherpa-onnx-vits','executable':'unused-ui-fixture','model':str(model),'tokens':str(tokens),'provider':'cpu'}))
        with (root/'runner.log').open('w+') as log:
            runner=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/test-desktop-ui.py'),'--config',str(config.resolve()),'--bin-dir',str(args.bin_dir.resolve()),'--script',str(script),'--pointer-target',str(target),'--timeout','60'],stdout=log,stderr=subprocess.STDOUT)
            try:
                # Include the service startup allowance and the child's UI
                # deadline, so a failed webview returns its diagnostic logs.
                deadline=time.monotonic()+80
                while not target.exists():
                    assert runner.poll() is None,'native driver exited before readiness'
                    assert time.monotonic()<deadline,'native menu readiness timed out'
                    time.sleep(.05)
                geometry=json.loads(target.read_text());window=None;deadline=time.monotonic()+5
                while window is None and time.monotonic()<deadline:
                    for candidate in visible():
                        result=xdo('getwindowgeometry','--shell',candidate)
                        properties=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
                        if properties.get('WIDTH')==str(round(geometry['width']*geometry['scale'])) and properties.get('HEIGHT')==str(round(geometry['height']*geometry['scale'])):
                            window=candidate;break
                    if window is None:time.sleep(.05)
                assert window is not None,'native player window not found'
                before=visible()
                assert xdo('mousemove','--window',window,round(geometry['x']*geometry['scale']),round(geometry['y']*geometry['scale']),'click','1').returncode==0
                time.sleep(.75);opened=visible()
                assert window in opened,'player dismissed when its native menu opened'
                assert opened-before,'no native popup appeared after mouse input'
                assert xdo('mousemove','--window',window,100,100,'click','1').returncode==0
                time.sleep(.5);assert window in visible(),'player dismissed while closing its own menu'
                time.sleep(4)
                assert xdo('mousemove','0','0','click','1').returncode==0
                assert runner.wait(timeout=20)==0,'native menu acceptance failed'
                log.flush();log.seek(0);print(log.read())
            except Exception:
                log.flush();log.seek(0);print(log.read());raise
            finally:
                if runner.poll() is None:
                    runner.send_signal(2)
                    try:runner.wait(timeout=15)
                    except subprocess.TimeoutExpired:runner.kill();runner.wait()


if __name__=='__main__':main()
