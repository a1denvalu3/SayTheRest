#!/usr/bin/env python3
"""Exercise all native Qwen language controls with local model assets and audio.

Checks valid nonempty WAVs and rejected unknown languages; it does not measure
intelligibility or pronunciation quality. No audio is uploaded.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import wave

SAMPLES = [
    ('auto','The morning is quiet and clear.'),
    ('en','The morning is quiet and clear.'),
    ('zh','早上好。今天的天气很好。'),
    ('ja','こんにちは。今日は良い天気です。'),
    ('ko','안녕하세요. 오늘 날씨가 좋습니다.'),
    ('de','Guten Morgen. Der Himmel ist klar.'),
    ('fr','Bonjour. La matinée est calme.'),
    ('ru','Доброе утро. Сегодня хорошая погода.'),
    ('pt','Bom dia. A manhã está tranquila.'),
    ('es','Buenos días. La mañana está tranquila.'),
    ('it','Buongiorno. La mattina è tranquilla.'),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True,help='Installed native ONNX Qwen EngineConfig JSON')
    parser.add_argument('--reference',type=Path,required=True,help='Local permitted reference WAV')
    parser.add_argument('--reference-text',help='Exact reference transcript; omit for embedding-only conditioning')
    parser.add_argument('--languages',help='Comma-separated subset; defaults to all ten languages and automatic detection')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    assert config['engine']=='qwen3-tts-onnx','native Qwen configuration required'
    runtime=Path(config['runtime']).resolve();env=dict(os.environ)
    key='PATH' if os.name=='nt' else 'LD_LIBRARY_PATH'
    env[key]=str(runtime.parent)+os.pathsep+env.get(key,'')
    chosen=set(args.languages.split(',')) if args.languages else {code for code,_ in SAMPLES}
    assert chosen and chosen<={code for code,_ in SAMPLES},'unknown requested test language'
    with tempfile.TemporaryDirectory(prefix='sayit-qwen-languages-') as directory:
        root=Path(directory)
        with (root/'worker.log').open('w+') as log:
            process=subprocess.Popen([str(Path(config['worker']).resolve()),'--model',str(Path(config['model']).resolve()),
                '--runtime',str(runtime),'--precision',config['precision'],'--device','cpu','--threads',str(config.get('num_threads',4))],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',env=env)
            try:
                assert json.loads(process.stdout.readline())['ready']
                def synthesize(code,text,output):
                    request={'text':text,'language':code,'output':str(output),'reference_audio':str(args.reference.resolve()),
                        'reference_text':args.reference_text,'seed':417,'max_new_tokens':256}
                    process.stdin.write(json.dumps(request)+'\n');process.stdin.flush()
                    return json.loads(process.stdout.readline())
                invalid=root/'invalid.wav'
                response=synthesize('unsupported','This must fail.',invalid)
                assert not response.get('ok') and 'language' in response.get('error','').lower(),response
                assert not invalid.exists(),'invalid language published audio'
                for code,text in SAMPLES:
                    if code not in chosen:continue
                    output=root/(code+'.wav');start=time.monotonic();response=synthesize(code,text,output)
                    assert response.get('ok'),response
                    with wave.open(str(output)) as wav:
                        seconds=wav.getnframes()/wav.getframerate()
                        assert wav.getnchannels()==1 and wav.getframerate()==24000 and seconds>0.5
                        assert any(wav.readframes(wav.getnframes())),'generated audio is silent'
                    print(json.dumps({'language':code,'passed':True,'generation_seconds':round(time.monotonic()-start,2),'audio_seconds':round(seconds,2)}),flush=True)
            except Exception:
                log.flush();log.seek(0);print(log.read()[-6000:]);raise
            finally:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
