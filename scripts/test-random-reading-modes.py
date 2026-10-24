#!/usr/bin/env python3
"""Real native Qwen random reading, per-paragraph chunks, cancellation and recovery.

Uses private state and silent device playback. --benchmark-only needs no audio
device and verifies native benchmarking with no recorded/saved voice profile.
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
import wave


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--bin-dir',type=Path,default=Path('target/release'))
    parser.add_argument('--benchmark-only',action='store_true')
    args=parser.parse_args()
    binary=args.bin_dir.resolve()/('sayit-service.exe' if os.name=='nt' else 'sayit-service')
    with tempfile.TemporaryDirectory(prefix='sayit-random-reading-') as temporary:
        root=Path(temporary)
        def free_port():
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]
        port=free_port();service=None
        with (root/'service.log').open('w+') as log:
            def until(predicate,timeout=240):
                end=time.monotonic()+timeout
                while time.monotonic()<end:
                    value=predicate()
                    if value:return value
                    assert service.poll() is None,'service exited'
                    time.sleep(1)
                raise AssertionError('random reading acceptance timed out')
            def start():
                nonlocal service
                service=subprocess.Popen([str(binary),'--port',str(port),'--data-dir',str(root),'--config',str(args.config.resolve())],stdout=log,stderr=subprocess.STDOUT)
                def healthy():
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/health',timeout=1):return True
                    except OSError:return False
                until(healthy,15)
            def stop():
                service.terminate()
                try:service.wait(timeout=10)
                except subprocess.TimeoutExpired:service.kill();service.wait()
            def api(resource,body=None,timeout=15):
                token_file='api-read-token' if body is None else 'api-token'
                request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/'+resource,data=None if body is None else json.dumps(body).encode(),
                    headers={'Authorization':'Bearer '+(root/token_file).read_text().strip(),'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=timeout) as response:
                    data=response.read();return json.loads(data) if data else None
            def finish(job):
                def generated():
                    current=next(item for item in api('jobs') if item['id']==job['id'])
                    assert current['state'] not in ['failed','cancelled'],current
                    playback=api('state')['playback']
                    return playback if playback['job_id']==job['id'] and playback['generation_complete'] and playback['duration_seconds']>0 else None
                playback=until(generated)
                assert playback['spoken_text']==job['text'],'preparation text leaked into the reader'
                assert any(playback['amplitudes']) and playback['spoken_chunks']
                api('playback/seek',{'seconds':playback['duration_seconds']-.1})
                item=until(lambda:next((item for item in api('history') if item['job']['id']==job['id']),None),30)
                assert item['job']['voice_mode']==job['voice_mode'] and item['job']['voice_profile_id'] is None
                with wave.open(item['audio_path']) as wav:
                    assert wav.getframerate()==24000 and wav.getnchannels()==1
                    assert abs(wav.getnframes()/wav.getframerate()-playback['duration_seconds'])<.001
                    assert any(wav.readframes(wav.getnframes()))
                return playback
            try:
                start();api('settings',{'volume':0,'compute_device':'cpu'})
                model=next(model for model in api('models') if model['selected']);model_id=model['id']
                assert model['capabilities']['random_voice_sampling'] and not model['requires_voice_profile']
                assert api('voices')==[]
                if args.benchmark_only:
                    report=api('models/'+model_id+'/benchmark',{},timeout=240)
                    assert report['iterations']==3 and report['audio_seconds']>0 and report['cold_start_ms']>0
                    assert api('voices')==[] and api('history')==[]
                    print(json.dumps({'passed':True,'benchmark_without_profile':True,'iterations':3,'audio_seconds':report['audio_seconds'],'cold_start_ms':round(report['cold_start_ms'])}));return
                first=api('jobs',{'text':'The morning is quiet and the words are clear.','queue_policy':'append'})
                assert first['voice_mode']=='automatic_stable' and first['voice_profile_id'] is None
                finish(first)
                api('models/'+model_id+'/voice-mode',{'mode':'random_per_paragraph'})
                text='The morning is quiet and the words are clear. '*4+'\nA new paragraph begins with a different voice.'
                second=api('jobs',{'text':text,'queue_policy':'append'})
                assert second['voice_mode']=='random_per_paragraph'
                api('models/'+model_id+'/voice-mode',{'mode':'automatic_stable'})
                playback=finish(second)
                assert len(playback['spoken_chunks'])>=3,'long paragraphs were not split into bounded blocks'
                assert all(chunk['text_end']-chunk['text_start']<=160 for chunk in playback['spoken_chunks'])
                cancelled=api('jobs',{'text':'This reading will be cancelled while preparing its voice.','queue_policy':'append'})
                until(lambda:any(job['id']==cancelled['id'] and job['state']=='synthesizing' for job in api('jobs')))
                time.sleep(2);cancel_start=time.monotonic();api('playback/stop',{})
                until(lambda:any(job['id']==cancelled['id'] and job['state']=='cancelled' for job in api('jobs')),10)
                assert not (root/'history'/f"{cancelled['id']}.wav").exists()
                assert not any(item['job']['id']==cancelled['id'] for item in api('history'))
                cancellation_seconds=time.monotonic()-cancel_start
                recovered=api('jobs',{'text':'A new voice can read successfully after cancellation.','queue_policy':'append'})
                finish(recovered);assert api('voices')==[],'reading modes created saved profiles'
                stop();port=free_port();start()
                history=api('history')
                assert next(item for item in history if item['job']['id']==second['id'])['job']['voice_mode']=='random_per_paragraph'
                assert api('settings')['voice_mode_by_model'][model_id]=='automatic_stable'
                print(json.dumps({'passed':True,'automatic_stable':True,'random_per_paragraph':True,'long_paragraph_blocks':len(playback['spoken_chunks']),'cancellation_seconds':round(cancellation_seconds,2),'recovery':True,'restart':True,'no_saved_profiles':True}))
            except Exception:
                log.flush();log.seek(0);print(log.read()[-7000:]);raise
            finally:
                if service is not None and service.poll() is None:stop()


if __name__=='__main__':main()
