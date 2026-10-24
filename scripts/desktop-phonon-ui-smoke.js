(async()=>{
  const invoke=window.__TAURI__.core.invoke;
  const get=resource=>invoke('service_get',{resource});
  const post=(resource,body={})=>invoke('service_post',{resource,body});
  const $=selector=>document.querySelector(selector),checks=[];
  const check=(condition,message)=>{if(!condition)throw Error(message);checks.push(message)};
  const until=async(predicate,timeout=45000)=>{
    const end=performance.now()+timeout;
    while(performance.now()<end){const value=await predicate();if(value)return value;await new Promise(resolve=>setTimeout(resolve,100));}
    throw Error('Phonon UI acceptance timed out');
  };
  try {
    await post('settings',{volume:0,compute_device:'cpu',speaking_pace:1.25});
    await invoke('open_management',{section:'voices'});
    await until(()=>!$('#voice-file-form').hidden&&document.querySelectorAll('[data-preset-action=preview]').length===38);
    check(getComputedStyle($('#voice-form')).display==='none'&&getComputedStyle($('#voice-file-form')).display!=='none','Phonon offers voice-file import with its 38 presets');
    check(getComputedStyle($('#voice-discovery')).display==='none','unsupported voice discovery stays hidden for Phonon');
    check(getComputedStyle($('#random-voice-modes')).display==='none','Qwen random reading modes stay hidden for Phonon');
    check(document.documentElement.scrollWidth<=innerWidth,'management layout has no horizontal overflow');
    $('[data-preset-voice=Freya][data-preset-action=select]').click();
    await until(async()=>(await get('models')).find(model=>model.selected)?.preset_voices.find(voice=>voice.id==='Freya')?.selected);
    check((await get('models')).find(model=>model.selected).preset_voices.filter(voice=>voice.selected).length===1,'preset selection is explicit and exclusive');
    const form=$('#voice-file-form');
    form.elements.name.value='Broken file';form.elements.path.value=window.__SAYIT_UI_FIXTURE__.bad_voice_file;
    form.requestSubmit();await until(()=>!form.querySelector('[type=submit]').disabled);
    check((await get('voices')).length===0,'invalid voice import leaves no profile');
    check($('#toast').textContent.includes('Phonon'),'invalid file gives an actionable error');
    form.elements.name.value='Imported narrator';form.elements.path.value=window.__SAYIT_UI_FIXTURE__.voice_file;
    form.requestSubmit();
    const profile=await until(async()=>(await get('voices')).find(voice=>voice.name==='Imported narrator'));
    await until(()=>$(`[data-voice="${profile.id}"][data-voice-action=select]`));
    check(profile.reference_kind==='phonon-embedding','native import persists the voice-file type');
    const card=$(`[data-voice="${profile.id}"]`).closest('article');
    check(card.textContent.includes('Phonon voice file')&&!card.querySelector('[data-voice-action=tune]'),'imported voice has file controls and metadata');
    $(`[data-voice="${profile.id}"][data-voice-action=select]`).click();
    await until(async()=>(await get('settings')).active_voice_profile_id===profile.id);
    await until(()=>$(`[data-voice="${profile.id}"][data-voice-action=preview]`));
    $(`[data-voice="${profile.id}"][data-voice-action=preview]`).click();
    const job=await until(async()=>(await get('jobs')).find(job=>job.voice_profile_id===profile.id));
    const playback=await until(async()=>{const p=(await get('state')).playback;return p.job_id===job.id&&p.generation_complete?p:null;});
    check(playback.amplitudes.some(value=>value>0)&&playback.duration_seconds>1,'imported voice produces real local speech');
    check(job.speaking_pace===1,'Phonon uses natural pace despite a different global model setting');
    await post('playback/seek',{seconds:playback.duration_seconds-0.2});
    await until(async()=>(await get('state')).playback.state==='completed');
    check((await get('history')).some(item=>item.job.id===job.id),'voice preview archives successfully');
    $('[data-preset-voice=default][data-preset-action=select]').click();
    await until(async()=>(await get('settings')).active_voice_profile_id===null);
    await until(()=>$(`[data-voice="${profile.id}"][data-voice-action=delete]`));
    $(`[data-voice="${profile.id}"][data-voice-action=delete]`).click();
    await until(async()=>(await get('voices')).length===0);
    check(true,'switching to a preset allows deleting the imported voice');
    check((await get('models')).find(model=>model.selected).capabilities.voice_file_import,'service advertises the tested import capability');
    await window.__TAURI__.event.emit('ui-test-result',{passed:true,checks});
  } catch(error) {
    await window.__TAURI__.event.emit('ui-test-result',{passed:false,error:String(error),stack:error.stack,toast:$('#toast')?.textContent,checks});
  }
})();
