#!/usr/bin/env python3
"""Real Kokoro acceptance for captured presets, overrides, history, and restart.

Uses private state and a real audio device at zero volume. The input config and
model assets are read only. --audio-output retains synthetic review samples.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--bin-dir',type=Path,default=Path('target/release'))
    parser.add_argument('--audio-output',type=Path)
    args=parser.parse_args()
    assert json.loads(args.config.read_text())['engine']=='sherpa-onnx-kokoro'
    binary=args.bin_dir.resolve()/('sayit-service.exe' if os.name=='nt' else 'sayit-service')
    with tempfile.TemporaryDirectory(prefix='sayit-queued-presets-') as temporary:
        root=Path(temporary);service=None;port=None
        with (root/'service.log').open('w+') as log:
            def until(predicate,timeout=180):
                end=time.monotonic()+timeout
                while time.monotonic()<end:
                    value=predicate()
                    if value:return value
                    assert service.poll() is None,'service exited'
                    time.sleep(1)
                raise AssertionError('queued-preset acceptance timed out')
            def start():
                nonlocal service,port
                with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
                service=subprocess.Popen([str(binary),'--data-dir',str(root),'--port',str(port),'--config',str(args.config.resolve())],stdout=log,stderr=subprocess.STDOUT)
                def healthy():
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/health',timeout=1):return True
                    except OSError:return False
                until(healthy,15)
            def stop():
                service.terminate()
                try:service.wait(timeout=10)
                except subprocess.TimeoutExpired:service.kill();service.wait()
            def api(resource,body=None,expected_error=None):
                token='api-read-token' if body is None else 'api-token'
                request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/'+resource,data=None if body is None else json.dumps(body).encode(),
                    headers={'Authorization':'Bearer '+(root/token).read_text().strip(),'Content-Type':'application/json'})
                try:
                    with urllib.request.urlopen(request,timeout=20) as response:
                        assert expected_error is None,'invalid preset was accepted'
                        data=response.read();return json.loads(data) if data else None
                except urllib.error.HTTPError as error:
                    detail=error.read().decode();assert error.code==expected_error,f'{error.code}: {detail}'
            def selected():return next(voice['id'] for voice in next(model for model in api('models') if model['selected'])['preset_voices'] if voice['selected'])
            def finish(job):
                def generated():
                    current=next(item for item in api('jobs') if item['id']==job['id'])
                    assert current['state'] not in ['failed','cancelled'],current
                    playback=api('state')['playback']
                    return playback if playback['job_id']==job['id'] and playback['generation_complete'] and playback['duration_seconds']>0 else None
                playback=until(generated)
                assert playback['spoken_text']==job['text'] and any(playback['amplitudes'])
                api('playback/seek',{'seconds':playback['duration_seconds']-.1})
                if playback['state']=='paused':api('playback/resume',{})
                item=until(lambda:next((item for item in api('history') if item['job']['id']==job['id']),None),30)
                assert item['job']['preset_voice_id']==job['preset_voice_id'] and item['job']['voice_profile_id'] is None
                with wave.open(item['audio_path']) as wav:
                    assert wav.getframerate()==24000 and wav.getnframes()>12000 and any(wav.readframes(wav.getnframes()))
                if args.audio_output:
                    args.audio_output.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(item['audio_path'],args.audio_output/(job['preset_voice_id']+'-'+job['id']+'.wav'))
            try:
                start();api('settings',{'volume':0,'compute_device':'cpu'})
                model='kokoro-int8-multi-lang-v1-1';api('models/'+model+'/voices/af_maple/select',{})
                first=api('jobs',{'text':'The morning light is clear and the words are easy to follow. '*4,'queue_policy':'append'})
                assert first['preset_voice_id']=='af_maple'
                until(lambda:api('state')['playback']['duration_seconds']>0);api('playback/pause',{})
                second=api('jobs',{'text':'This queued reading keeps its original voice.','queue_policy':'append'})
                assert second['preset_voice_id']=='af_maple'
                api('models/'+model+'/voices/bf_vale/select',{})
                for invalid in [
                    {'preset_voice_id':'unknown-preset'},
                    {'preset_voice_id':'af_maple','voice_profile_id':str(uuid.uuid4())},
                    {'preset_voice_id':'af_maple','voice_mode':'automatic_stable'},
                ]:
                    api('jobs',{'text':'This invalid override must not interrupt playback.','queue_policy':'interrupt',**invalid},expected_error=422)
                    assert api('state')['playback']['state']=='paused'
                    assert next(item for item in api('jobs') if item['id']==second['id'])['state']=='queued'
                third=api('jobs',{'text':'This request chooses a preset without changing the default.','preset_voice_id':'af_sol','queue_policy':'append'})
                assert third['preset_voice_id']=='af_sol' and selected()=='bf_vale'
                for job in [first,second,third]:finish(job)
                default=api('jobs',{'text':'This sentence uses the newly selected default voice.','queue_policy':'append'})
                assert default['preset_voice_id']=='bf_vale';finish(default)
                regenerated=api('history/'+second['id']+'/regenerate',{})
                assert regenerated['preset_voice_id']=='af_maple' and selected()=='bf_vale';finish(regenerated)
                stop();start();history=api('history')
                for job in [first,second,third,default,regenerated]:
                    assert next(item for item in history if item['job']['id']==job['id'])['job']['preset_voice_id']==job['preset_voice_id']
                assert selected()=='bf_vale'
                print(json.dumps({'passed':True,'captured_preset':'af_maple','per_request_override':'af_sol','selected_default':'bf_vale','history_regeneration':True,'restart':True,'invalid_interrupt_preserves_playback':True}))
            except Exception:
                log.flush();log.seek(0);print(log.read()[-7000:]);raise
            finally:
                if service is not None and service.poll() is None:stop()


if __name__=='__main__':main()
