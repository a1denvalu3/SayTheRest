#!/usr/bin/env python3
"""Verify Qwen language, cloning, pace, history, and silent device playback.

Uses local assets through an isolated service; requires an available audio device.
"""
import argparse
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
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--reference-text',required=True,help='Exact words in the permitted reference WAV')
    parser.add_argument('--bin-dir',type=Path,default=Path('target/debug'))
    parser.add_argument('--language',default='es')
    parser.add_argument('--text',default='Buenos días. La mañana está tranquila.')
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    binary=args.bin_dir.resolve()/('sayit-service.exe' if os.name=='nt' else 'sayit-service')
    with tempfile.TemporaryDirectory(prefix='sayit-qwen-service-') as temporary:
        root=Path(temporary)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with (root/'service.log').open('w+') as log:
            service=subprocess.Popen([str(binary),'--data-dir',str(root),'--port',str(port),'--config',str(args.config.resolve())],stdout=log,stderr=subprocess.STDOUT)
            try:
                end=time.monotonic()+15
                while not (root/'api-token').exists():
                    assert service.poll() is None,'service exited'
                    assert time.monotonic()<end,'service startup timed out';time.sleep(.05)
                token=(root/'api-token').read_text().strip()
                def api(resource,body=None):
                    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/'+resource,
                        data=None if body is None else json.dumps(body).encode(),
                        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=15) as response:
                        data=response.read();return json.loads(data) if data else None
                settings=api('settings',{'volume':0,'compute_device':'cpu','speaking_pace':1.25})
                model=settings['active_model_id'];assert model
                api(f'models/{model}/language',{'language':args.language})
                voice=api('voices',{'name':'Language acceptance','reference_audio_path':str(args.reference.resolve()),
                    'reference_text':args.reference_text,'speaker_permission_confirmed':True})
                api('voices/'+voice['id']+'/select',{})
                job=api('jobs',{'text':args.text,'queue_policy':'append'})
                assert job['language']==args.language and job['speaking_pace']==1,job
                start=time.monotonic();end=start+args.timeout
                while True:
                    state=api('state');p=state['playback'];jobs=api('jobs')
                    current=next(item for item in jobs if item['id']==job['id'])
                    assert current['state']!='failed',current
                    if p['job_id']==job['id'] and p['generation_complete']:
                        assert p['duration_seconds']>0.5 and any(p['amplitudes'])
                        api('playback/seek',{'seconds':max(0,p['duration_seconds']-.1)});break
                    assert time.monotonic()<end,'Qwen synthesis timed out';time.sleep(1)
                while True:
                    item=next((item for item in api('history') if item['job']['id']==job['id']),None)
                    if item:
                        assert item['job']['state']=='completed' and item['job']['language']==args.language;break
                    assert time.monotonic()<end,'Qwen history publication timed out';time.sleep(.5)
                assert state['inference_device']['provider']=='cpu',state['inference_device']
                print(json.dumps({'passed':True,'backend':state['inference_device']['backend'],'language':job['language'],
                    'natural_pace':True,'acceptance_seconds':round(time.monotonic()-start,2),'audio_seconds':round(p['duration_seconds'],2)}))
            except Exception:
                log.flush();log.seek(0);print(log.read()[-6000:]);raise
            finally:
                service.terminate()
                try:service.wait(timeout=10)
                except subprocess.TimeoutExpired:service.kill();service.wait()


if __name__=='__main__':main()
