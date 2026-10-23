#!/usr/bin/env python3
"""Exercise every pinned Phonon preset against the native worker."""
import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import wave


def pcm16(value):
    value=struct.unpack('<f',struct.pack('<f',value))[0]
    value=struct.unpack('<f',struct.pack('<f',max(-1.0,min(1.0,value))*32767))[0]
    return math.floor(value+0.5) if value>=0 else math.ceil(value-0.5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--formats-only',action='store_true',help='Check only the imported F32/F16/BF16 voice formats')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    pack=json.loads((repo/'crates/core/data/phonon-voices.json').read_text())
    voices=['default']+[voice['id'] for voice in pack['voices']]

    def check_voice(voice,reference=None):
        with tempfile.TemporaryDirectory(prefix='sayit-phonon-voice-') as directory:
            root=Path(directory)
            config=root/'config.json';output=root/'private.wav'
            config.write_text(json.dumps({'engine':'phonon','model':str(args.model.resolve()),'worker':str(args.worker.resolve()),'voice':voice}))
            request={'text':'Hello from this voice.','output':str(output),'stream':True,'speaking_pace':1.0}
            if reference:request['reference_audio']=str(reference)
            result=subprocess.run([str(args.worker.resolve()),'--phonon-config',str(config)],input=json.dumps(request)+'\n',
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120)
            assert result.returncode==0,(voice,result.stderr[-2000:])
            messages=[json.loads(line) for line in result.stdout.splitlines()]
            assert messages[0].get('ready') is True,(voice,messages[0])
            assert messages[0].get('backend')=='phonon'
            assert messages[-1].get('ok') is True,(voice,messages[-1])
            blocks=[message['audio'] for message in messages if 'audio' in message]
            assert len(blocks)>1,(voice,'expected incremental audio')
            assert all(block['sample_rate']==24000 for block in blocks),voice
            samples=[value for block in blocks for value in block['samples']]
            assert all(math.isfinite(value) for value in samples),voice
            assert any(abs(value)>0.01 for value in samples),voice
            with wave.open(str(output)) as audio:
                assert audio.getframerate()==24000 and audio.getnchannels()==1 and audio.getsampwidth()==2,voice
                assert audio.getnframes()==len(samples),voice
                archive=audio.readframes(audio.getnframes())
            expected=struct.pack('<'+'h'*len(samples),*(pcm16(value) for value in samples))
            assert archive==expected,(voice,'streamed samples differ from the archive')
            return {'voice':voice,'reference':reference.name if reference else None,'samples':len(samples),'blocks':len(blocks)}

    if not args.formats_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(check_voice,voices):
                print(json.dumps(result),flush=True)
    original=(args.model/'default-voice.safetensors').read_bytes()
    header_size=struct.unpack('<Q',original[:8])[0]
    header=json.loads(original[8:8+header_size])
    values=[value[0] for value in struct.iter_unpack('<f',original[8+header_size:])]
    with tempfile.TemporaryDirectory(prefix='sayit-phonon-formats-') as directory:
        for dtype in ['F32','F16','BF16']:
            if dtype=='F32':data=original[8+header_size:]
            elif dtype=='F16':data=b''.join(struct.pack('<e',value) for value in values)
            else:
                bits=[struct.unpack('<I',struct.pack('<f',value))[0] for value in values]
                data=b''.join(struct.pack('<H',((value+0x7fff+((value>>16)&1))>>16)&0xffff) for value in bits)
            header['speaker_wavs'].update(dtype=dtype,data_offsets=[0,len(data)])
            encoded=json.dumps(header).encode()
            reference=Path(directory)/f'{dtype}.safetensors'
            reference.write_bytes(struct.pack('<Q',len(encoded))+encoded+data)
            print(json.dumps(check_voice('default',reference)),flush=True)
    print(json.dumps({'passed':True,'voices':0 if args.formats_only else len(voices),'import_formats':['F32','F16','BF16']}))


if __name__=='__main__':main()
