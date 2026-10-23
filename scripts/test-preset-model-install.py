#!/usr/bin/env python3
"""Install, select, update, and optionally play a preset model in an isolated service."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='phonon-31m-en')
    parser.add_argument('--bin-dir',type=Path,default=Path('target/release'))
    parser.add_argument('--install-only',action='store_true',help='Do not open an audio device')
    args=parser.parse_args()
    binary=args.bin_dir.resolve()/('sayit-service.exe' if os.name=='nt' else 'sayit-service')
    with tempfile.TemporaryDirectory(prefix='sayit-model-install-') as directory:
        root=Path(directory)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with (root/'service.log').open('w+') as log:
            process=subprocess.Popen([str(binary),'--data-dir',str(root),'--port',str(port)],stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+15
                while not (root/'api-token').exists():
                    assert process.poll() is None,'service exited'
                    assert time.monotonic()<deadline,'service startup timed out'
                    time.sleep(0.05)
                token=(root/'api-token').read_text().strip()
                def api(resource,body=None):
                    request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/'+resource,data=None if body is None else json.dumps(body).encode(),
                        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=15) as response:
                        data=response.read();return json.loads(data) if data else None
                def until(predicate,timeout=45,interval=0.5):
                    deadline=time.monotonic()+timeout
                    while time.monotonic()<deadline:
                        value=predicate()
                        if value:return value
                        assert process.poll() is None,'service exited'
                        time.sleep(interval)
                    raise AssertionError('model acceptance timed out')
                until(lambda:api('health'))
                model=lambda:next(item for item in api('models') if item['id']==args.model)
                assert model()['capabilities']['preset_voices'],'this runner requires a model with presets'
                api('settings',{'volume':0,'compute_device':'cpu'})
                def installed():
                    item=model();download=item.get('download') or {}
                    assert download.get('state')!='failed',download.get('error')
                    return item if item['installed'] and download.get('state')=='installed' else None
                start=time.monotonic();api(f'models/{args.model}/install',{})
                item=until(installed,timeout=600,interval=1)
                first_install=time.monotonic()-start
                assert not item['update_available']
                api(f'models/{args.model}/select',{})
                presets=model()['preset_voices'];assert presets
                chosen=next((voice for voice in presets if voice['id']=='Freya'),presets[-1])
                api(f"models/{args.model}/voices/{chosen['id']}/select",{})
                assert next(voice for voice in model()['preset_voices'] if voice['id']==chosen['id'])['selected']
                manifest=root/'models'/args.model/'catalog-install.json';previous=manifest.stat().st_mtime_ns
                start=time.monotonic();api(f'models/{args.model}/update',{})
                def updated():
                    try:changed=manifest.stat().st_mtime_ns!=previous
                    except FileNotFoundError:return False
                    return changed and installed()
                until(updated,timeout=600,interval=1)
                update=time.monotonic()-start
                assert next(voice for voice in model()['preset_voices'] if voice['id']==chosen['id'])['selected'],'update lost the selected voice'
                revision_reload=None
                if not args.install_only:
                    def speak():
                        job=api('jobs',{'text':'The morning is quiet and the words are clear.','queue_policy':'append'})
                        playback=until(lambda:(p if (p:=api('state')['playback'])['job_id']==job['id'] and p['generation_complete'] else None),timeout=90)
                        assert playback['duration_seconds']>1 and any(playback['amplitudes'])
                        api('playback/seek',{'seconds':playback['duration_seconds']-0.2})
                        history=until(lambda:next((entry for entry in api('history') if entry['job']['id']==job['id']),None))
                        assert history['job']['state']=='completed'
                        assert history['spoken_chunks']
                        return history
                    history=speak()
                    if args.model=='phonon-31m-en':
                        # Simulate a new voice-pack revision at the same paths.
                        # A resident worker must reload instead of serving the
                        # previous embedding from memory.
                        before=hashlib.sha256(Path(history['audio_path']).read_bytes()).digest()
                        voices=root/'models'/args.model/'voices'
                        (voices/(chosen['id']+'.safetensors')).write_bytes((voices/'Zoey.safetensors').read_bytes())
                        metadata=json.loads(manifest.read_text());metadata['sha256']='acceptance-new-voice-revision'
                        manifest.write_text(json.dumps(metadata))
                        after=speak()
                        assert hashlib.sha256(Path(after['audio_path']).read_bytes()).digest()!=before,'new voice-pack revision reused stale model memory'
                        revision_reload=True
                print(json.dumps({'passed':True,'model':args.model,'presets':len(presets),'install_seconds':round(first_install,2),
                    'update_seconds':round(update,2),'device_playback':not args.install_only,'revision_reload':revision_reload}))
            except Exception:
                log.flush();log.seek(0);print(log.read()[-6000:]);raise
            finally:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
