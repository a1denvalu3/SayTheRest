(async()=>{
  const invoke=window.__TAURI__.core.invoke,get=resource=>invoke('service_get',{resource}),post=(resource,body={})=>invoke('service_post',{resource,body});
  const $=selector=>document.querySelector(selector),checks=[];
  const check=(value,message)=>{if(!value)throw Error(message);checks.push(message)};
  const until=async(predicate,timeout=90000)=>{const end=performance.now()+timeout;while(performance.now()<end){const value=await predicate();if(value)return value;await new Promise(resolve=>setTimeout(resolve,2000));}throw Error('Kokoro UI acceptance timed out')};
  let previous=null;
  async function preview(id,expected){
    const button=$(`[data-preset-voice="${id}"][data-preset-action=preview]`);button.scrollIntoView({block:'center'});button.click();
    const p=await until(async()=>{const p=(await get('state')).playback;return p.job_id&&p.job_id!==previous&&p.generation_complete&&p.duration_seconds>0?p:null});
    check(p.amplitudes.some(value=>value>0)&&p.spoken_text.includes(expected),`${id} preview produces actual speech with its matching reader passage`);
    previous=p.job_id;await post('playback/seek',{seconds:p.duration_seconds-.1});await until(async()=>(await get('state')).playback.state==='completed');
  }
  try{
    await post('settings',{volume:0,compute_device:'cpu'});await invoke('open_management',{section:'voices'});
    await until(()=>document.querySelectorAll('[data-preset-action=preview]').length===103);
    const model=(await get('models')).find(model=>model.selected);
    check(model.id==='kokoro-int8-multi-lang-v1-1'&&model.preset_voices.length===103,'Kokoro offers the installed pack of 103 presets');
    check(model.preset_voices.find(voice=>voice.selected)?.id==='af_maple','fresh English Kokoro configuration selects Maple');
    check(model.languages.join(',')==='en-US,en-GB,zh-CN'&&model.quality_note.includes('103'),'Kokoro advertises its actual voice count and supported locales');
    check(getComputedStyle($('#voice-form')).display==='none'&&getComputedStyle($('#voice-file-form')).display==='none'&&getComputedStyle($('#random-voice-modes')).display==='none','Kokoro hides unsupported cloning, voice files and Qwen modes');
    check(document.documentElement.scrollWidth<=innerWidth,'Kokoro voice cards fit the native management window');
    await preview('af_maple','Maple');
    for(const [id,language,passage] of [['bf_vale','en-GB','Vale'],['zf_001','zh-CN','你好，这是声音试听。']]){
      const select=$(`[data-preset-voice="${id}"][data-preset-action=select]`);select.scrollIntoView({block:'center'});select.click();
      await until(async()=>(await get('models')).find(model=>model.selected).preset_voices.find(voice=>voice.id===id)?.selected);
      check((await get('models')).find(model=>model.selected).preset_voices.find(voice=>voice.selected).language===language,`${id} selection reports its matching locale`);
      await preview(id,passage);
    }
    check((await get('history')).length===0&&(await get('voices')).length===0,'Kokoro previews add neither history nor saved profiles');
    await window.__TAURI__.event.emit('ui-test-result',{passed:true,checks});
  }catch(error){await window.__TAURI__.event.emit('ui-test-result',{passed:false,error:String(error),stack:error.stack,toast:$('#toast')?.textContent,checks})}
})();
