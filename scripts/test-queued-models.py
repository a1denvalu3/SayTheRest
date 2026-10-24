#!/usr/bin/env python3
"""Verify queued-model capture and history regeneration with real Piper/Phonon audio.

Uses private service state and a real audio device at zero volume. The supplied
configurations and their model assets are only read.
"""
import argparse
import json
import os
from pathlib import Path
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
    parser.add_argument('--piper-config',type=Path,required=True)
    parser.add_argument('--phonon-config',type=Path,required=True)
    parser.add_argument('--bin-dir',type=Path,default=Path('target/release'))
    args=parser.parse_args()
    assert json.loads(args.piper_config.read_text())['engine']=='sherpa-onnx-vits'
    assert json.loads(args.phonon_config.read_text())['engine']=='phonon'
    binary=args.bin_dir.resolve()/('sayit-service.exe' if os.name=='nt' else 'sayit-service')
    with tempfile.TemporaryDirectory(prefix='sayit-queued-models-') as temporary:
        root=Path(temporary);other=root/'models/phonon-31m-en/config.json'
        other.parent.mkdir(parents=True);other.write_text(args.phonon_config.read_text())
        with (root/'service.log').open('w+') as log:
            service=None;port=None
            def until(predicate,timeout=60):
                end=time.monotonic()+timeout
                while time.monotonic()<end:
                    value=predicate()
                    if value:return value
                    assert service.poll() is None,'service exited'
                    time.sleep(.1)
                raise AssertionError('queued-model acceptance timed out')
            def start():
                nonlocal service,port
                with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
                service=subprocess.Popen([str(binary),'--data-dir',str(root),'--port',str(port),'--config',str(args.piper_config.resolve())],stdout=log,stderr=subprocess.STDOUT)
                def healthy():
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/health',timeout=1):return True
                    except OSError:return False
                until(healthy,15)
            def stop():
                service.terminate()
                try:service.wait(timeout=10)
                except subprocess.TimeoutExpired:service.kill();service.wait()
            def api(resource,body=None,expected_error=None,method=None):
                request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/'+resource,data=None if body is None else json.dumps(body).encode(),
                    method=method,headers={'Authorization':'Bearer '+(root/'api-token').read_text().strip(),'Content-Type':'application/json'})
                try:
                    with urllib.request.urlopen(request,timeout=15) as response:
                        assert expected_error is None,'invalid submission was accepted'
                        data=response.read();return json.loads(data) if data else None
                except urllib.error.HTTPError as error:
                    detail=error.read().decode()
                    assert error.code==expected_error,f'{resource}: {error.code} {detail}'
            def finish(job,expected_rate):
                def generated():
                    current=next(item for item in api('jobs') if item['id']==job['id'])
                    assert current['state'] not in ['failed','cancelled'],current
                    playback=api('state')['playback']
                    return playback if playback['job_id']==job['id'] and playback['generation_complete'] and playback['duration_seconds']>0 else None
                playback=until(generated)
                assert any(playback['amplitudes']),'audio contains no speech amplitudes'
                api('playback/seek',{'seconds':playback['duration_seconds']-.1})
                if playback['state']=='paused':api('playback/resume',{})
                item=until(lambda:next((item for item in api('history') if item['job']['id']==job['id']),None))
                with wave.open(item['audio_path']) as wav:
                    assert wav.getframerate()==expected_rate,(job['model_id'],wav.getframerate(),expected_rate)
                    assert wav.getnframes()>expected_rate*.5 and any(wav.readframes(wav.getnframes()))
                assert item['job']['model_id']==job['model_id']
            try:
                start();api('settings',{'volume':0,'compute_device':'cpu'})
                piper='piper-en-us-lessac-medium';phonon='phonon-31m-en'
                first=api('jobs',{'text':'The morning light is clear and the words are easy to follow. '*6,'queue_policy':'append'})
                until(lambda:api('state')['playback']['duration_seconds']>0)
                api('playback/pause',{})
                second=api('jobs',{'text':'This queued reading keeps the engine it was given.','queue_policy':'append'})
                assert first['model_id']==second['model_id']==piper
                api('models/'+phonon+'/select',{})
                api('models/'+piper,expected_error=409,method='DELETE')
                api('jobs',{'text':'This request must fail without interrupting speech.','voice_profile_id':str(uuid.uuid4()),'queue_policy':'interrupt'},expected_error=409)
                playback=api('state')['playback']
                assert playback['job_id']==first['id'] and playback['state']=='paused','rejected request interrupted valid playback'
                assert next(item for item in api('jobs') if item['id']==second['id'])['state']=='queued'
                third=api('jobs',{'text':'An explicit model override leaves the active choice unchanged.','model_id':piper,'queue_policy':'append'})
                assert third['model_id']==piper and api('settings')['active_model_id']==phonon
                for job in [first,second,third]:finish(job,22050)
                current=api('jobs',{'text':'This final sentence uses the selected Phonon engine.','queue_policy':'append'})
                assert current['model_id']==phonon;finish(current,24000)
                regenerated=api('history/'+second['id']+'/regenerate',{})
                assert regenerated['model_id']==piper and api('settings')['active_model_id']==phonon
                finish(regenerated,22050)
                stop();start()
                history=api('history')
                assert next(item for item in history if item['job']['id']==second['id'])['job']['model_id']==piper
                assert next(item for item in history if item['job']['id']==current['id'])['job']['model_id']==phonon
                print(json.dumps({'passed':True,'queued_model':piper,'selected_model':phonon,'piper_hz':22050,'phonon_hz':24000,'history_regeneration':True,'restart':True,'invalid_interrupt_preserves_playback':True}))
            except Exception:
                log.flush();log.seek(0);print(log.read()[-6000:]);raise
            finally:
                if service is not None and service.poll() is None:stop()


if __name__=='__main__':main()
