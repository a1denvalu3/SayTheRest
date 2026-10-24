#!/usr/bin/env python3
"""Real native Qwen discovery, preview, save, cancellation, and profile reuse.

All service state is private to this run. --headless skips audio-device checks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--bin-dir',type=Path,default=Path('target/release'))
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--candidates',type=int,choices=range(1,9),help='Override the default batch of four')
    parser.add_argument('--randomize',action='store_true',help='Use and verify a different tuning recipe for each candidate')
    parser.add_argument('--reference-output',type=Path,help='Keep the generated synthetic reference for further local model tests')
    args=parser.parse_args()
    binary=args.bin_dir.resolve()/('sayit-service.exe' if os.name=='nt' else 'sayit-service')
    with tempfile.TemporaryDirectory(prefix='sayit-discovery-test-') as temporary:
        root=Path(temporary)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with (root/'service.log').open('w+') as log:
            service=subprocess.Popen([str(binary),'--data-dir',str(root),'--port',str(port),'--config',str(args.config.resolve())],stdout=log,stderr=subprocess.STDOUT)
            try:
                end=time.monotonic()+15
                while not (root/'api-token').exists():
                    assert service.poll() is None;assert time.monotonic()<end;time.sleep(.05)
                token=(root/'api-token').read_text().strip()
                def api(resource,body=None,method=None):
                    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/'+resource,data=None if body is None else json.dumps(body).encode(),method=method,
                        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
                    try:
                        with urllib.request.urlopen(req,timeout=15) as response:
                            data=response.read();return json.loads(data) if data else None
                    except urllib.error.HTTPError as error:
                        raise AssertionError(f'{resource}: {error.code} {error.read().decode()}') from error
                def until(predicate,timeout=180):
                    end=time.monotonic()+timeout
                    while time.monotonic()<end:
                        result=predicate()
                        if result:return result
                        assert service.poll() is None,'service exited';time.sleep(.5)
                    raise AssertionError('discovery acceptance timed out')
                api('settings',{'volume':0,'compute_device':'cpu'})
                assert next(model for model in api('models') if model['selected'])['capabilities']['voice_discovery']
                assert api('voices/discovery') is None
                sample='A quiet morning brings a clear new sentence. Every voice gives familiar words its own color, character, and rhythm.'
                request={'sample_text':sample,'language':'en','seed':417,'randomize_tuning':args.randomize}
                if args.candidates is not None:request['candidate_count']=args.candidates
                expected=args.candidates or 4
                start=time.monotonic();api('voices/discovery',request)
                def ready():
                    snapshot=api('voices/discovery');assert snapshot['state']!='failed',snapshot
                    return snapshot if snapshot['state']=='ready' else None
                snapshot=until(ready,timeout=420)
                candidate=next((item for item in snapshot['candidates'] if item['save_error'] is None),None)
                assert candidate is not None,'no candidate passed reference-quality checks'
                assert snapshot['completed']==snapshot['total']==len(snapshot['candidates'])==expected
                assert [item['seed'] for item in snapshot['candidates']]==list(range(417,417+expected))
                assert len({item['id'] for item in snapshot['candidates']})==expected
                assert all(item['duration_seconds']>0.5 for item in snapshot['candidates'])
                if args.randomize and expected>1:
                    assert len({json.dumps(item['tuning'],sort_keys=True) for item in snapshot['candidates']})>1,'randomized tuning was identical for every candidate'
                assert candidate['duration_seconds']>3
                assert candidate['save_error'] is None,candidate
                assert api('history')==[],'discovery created speech history'
                if not args.headless:
                    api(f"voices/discovery/{candidate['id']}/preview",{})
                    playback=api('state')['playback'];assert playback['spoken_text']==sample and any(playback['amplitudes'])
                    api('playback/seek',{'seconds':playback['duration_seconds']-.1})
                    until(lambda:api('state')['playback']['state']=='completed')
                path=f"voices/discovery/{candidate['id']}/save"
                with ThreadPoolExecutor(max_workers=2) as pool:
                    saved=list(pool.map(lambda _:api(path,{'name':'Generated narrator'}),range(2)))
                profile=saved[0];assert saved[1]['id']==profile['id'],'concurrent save created duplicate profiles'
                assert profile['generated'] and profile['language']=='en' and profile['generation_tuning']==candidate['tuning']
                assert len(api('voices'))==1 and Path(profile['reference_audio_path']).is_file()
                if args.reference_output:
                    args.reference_output.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(profile['reference_audio_path'],args.reference_output)
                api('voices/'+profile['id']+'/select',{})
                if not args.headless:
                    job=api('jobs',{'text':'The morning is quiet and the words are clear.','queue_policy':'append'})
                    def generated():
                        current=next(item for item in api('jobs') if item['id']==job['id']);assert current['state']!='failed',current
                        playback=api('state')['playback']
                        return playback if playback['job_id']==job['id'] and playback['generation_complete'] else None
                    playback=until(generated);assert any(playback['amplitudes'])
                    api('playback/seek',{'seconds':playback['duration_seconds']-.1})
                    until(lambda:any(item['job']['id']==job['id'] for item in api('history')))
                api('voices/discovery',{'sample_text':sample,'language':'en','candidate_count':2})
                time.sleep(2) # Let the real worker start inference before cancellation.
                cancel_start=time.monotonic()
                api('voices/discovery/cancel',{})
                cancelled=until(lambda:(snapshot if (snapshot:=api('voices/discovery'))['state']=='cancelled' else None))
                cancellation_seconds=time.monotonic()-cancel_start
                assert cancellation_seconds<10,'cancellation did not interrupt the worker promptly'
                assert any(item['id']==candidate['id'] for item in cancelled['candidates']),'cancellation discarded completed candidates'
                api('voices/discovery',method='DELETE')
                assert api('voices/discovery') is None
                assert Path(profile['reference_audio_path']).is_file(),'clearing samples deleted the saved profile'
                print(json.dumps({'passed':True,'model':snapshot['model_id'],'candidates':len(snapshot['candidates']),'device_playback':not args.headless,'profile_reuse':not args.headless,'concurrent_save':True,'cancellation_seconds':round(cancellation_seconds,2),'acceptance_seconds':round(time.monotonic()-start,2)}))
            except Exception:
                log.flush();log.seek(0);print(log.read()[-6000:]);raise
            finally:
                service.terminate()
                try:service.wait(timeout=10)
                except subprocess.TimeoutExpired:service.kill();service.wait()


if __name__=='__main__':main()
