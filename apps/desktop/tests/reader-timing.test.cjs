const {test}=require('node:test');
const assert=require('node:assert/strict');
const timing=require('../ui/reader-timing.js');

test('Unicode offsets agree with Rust scalar offsets and keep CJK words navigable',()=>{
  const text='Hello 🙂 日本語の文章。 Two words.';
  const characters=Array.from(text),words=timing.tokenize(text);
  for(const word of words) assert.equal(characters.slice(word.start,word.end).join(''),word.text);
  assert.equal(words.find(word=>word.text==='Two').start,16);
  assert.ok(words.filter(word=>/[\p{Script=Han}\p{Script=Hiragana}]/u.test(word.text)).length>1);
});

test('word seek positions follow measured block boundaries and paragraph gaps',()=>{
  const words=timing.tokenize('One two. Next words.');
  const chunks=[{text_start:0,text_end:8,audio_start_seconds:0,audio_end_seconds:2},
    {text_start:9,text_end:20,audio_start_seconds:3,audio_end_seconds:5.2}];
  const times=timing.timings(words,chunks,5.2,true);
  assert.deepEqual(times,[0,1,3,4]);
  assert.equal(timing.activeWord(times,1.2,5.2,chunks),1);
  assert.equal(timing.activeWord(times,2.5,5.2,chunks),-1);
  assert.equal(timing.activeWord(times,4.2,5.2,chunks),3);
  assert.equal(timing.activeWord(times,5.2,5.2,chunks),-1);
});

test('a growing live stream cannot move previous words or expose future blocks',()=>{
  const words=timing.tokenize('One two. Next words. Later words.');
  const chunks=[{text_start:0,text_end:8,audio_start_seconds:0,audio_end_seconds:2},
    {text_start:9,text_end:20,audio_start_seconds:2,audio_end_seconds:null}];
  const early=timing.timings(words,chunks,2.2,false),later=timing.timings(words,chunks,3.5,false);
  assert.deepEqual(early,later);
  assert.deepEqual(early.slice(0,2),[0,1]);
  assert.equal(early.at(-1),Infinity);
  assert.equal(timing.activeWord(early,2.2,2.2,chunks),-1);
});

test('legacy recordings get fixed-duration timing while empty live streams stay unseekable',()=>{
  const words=timing.tokenize('One two');
  assert.deepEqual(timing.timings(words,[],2,false),[Infinity,Infinity]);
  assert.deepEqual(timing.timings(words,[],7,true),[0,4]);
  assert.equal(timing.activeWord([],1,5),-1);
  assert.equal(timing.activeWord([0],NaN,5),-1);
});

test('total duration reflects source generation and stays independent of playback speed',()=>{
  assert.equal(timing.estimatedDuration('One two',[],7,true),7);
  assert.ok(timing.estimatedDuration('Words '.repeat(20),[],1,false)>1);
  assert.equal(timing.estimatedDuration('',[],0,false),0);
});
