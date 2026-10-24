(async()=>{
  const invoke=window.__TAURI__.core.invoke;
  const get=resource=>invoke('service_get',{resource});
  const post=(resource,body={})=>invoke('service_post',{resource,body});
  const $=selector=>document.querySelector(selector),checks=[];
  const check=(condition,message)=>{if(!condition)throw Error(message);checks.push(message)};
  const until=async(predicate,timeout=45000,interval=500)=>{
    const end=performance.now()+timeout;
    while(performance.now()<end){const value=await predicate();if(value)return value;await new Promise(resolve=>setTimeout(resolve,interval));}
    throw Error('Pocket UI acceptance timed out');
  };
  async function finish(jobId) {
    const p=await until(async()=>{const p=(await get('state')).playback;return p.job_id&&(!jobId||p.job_id===jobId)&&p.generation_complete&&p.duration_seconds>0?p:null;});
    check(p.amplitudes.some(value=>value>0),'voice produces real speech amplitudes');
    await post('playback/seek',{seconds:p.duration_seconds-0.1});
    await until(async()=>(await get('state')).playback.state==='completed');
    return p;
  }
  try {
    await post('settings',{volume:0,compute_device:'cpu',speaking_pace:1.25});
    await invoke('open_management',{section:'models'});
    await until(()=>$('[data-model=pocket-tts-int8][data-action=install]'));
    $('[data-model=pocket-tts-int8][data-action=install]').click();
    await until(async()=>{const m=(await get('models')).find(model=>model.id==='pocket-tts-int8');if(m.download?.state==='failed')throw Error(m.download.error);return m.installed&&m.download?.state==='installed';},600000,1000);
    await until(()=>$('[data-model=pocket-tts-int8][data-action=select]'));
    $('[data-model=pocket-tts-int8][data-action=select]').click();
    const selected=await until(async()=>(await get('models')).find(model=>model.id==='pocket-tts-int8'&&model.selected));
    check(!selected.requires_voice_profile&&selected.preset_voices.length===8,'installed Pocket offers eight immediately usable presets');
    await until(()=>$('#models-view').classList.contains('active'));
    check((await get('settings')).active_voice_profile_id===null,'selecting Pocket does not force a cloned profile');
    const job=await post('jobs',{text:'The morning is quiet and the words are clear.',queue_policy:'append'});
    check(job.voice_profile_id===null&&job.speaking_pace===1,'default preset reads with natural pace and no profile');
    await finish(job.id);
    const historyCount=(await get('history')).length;
    await invoke('open_management',{section:'voices'});
    await until(()=>document.querySelectorAll('[data-preset-action=preview]').length===8);
    check(getComputedStyle($('#voice-form')).display!=='none'&&getComputedStyle($('#voice-file-form')).display==='none','Pocket keeps optional audio cloning available');
    check(getComputedStyle($('#voice-discovery')).display==='none','unsupported voice discovery stays hidden for Pocket');
    check($('[data-preset-voice=cosette]').closest('article').textContent.includes('CC-BY-NC-4.0'),'preset recording license is visible');
    $('[data-preset-voice=cosette][data-preset-action=preview]').click();
    await until(async()=>{const p=(await get('state')).playback;return p.job_id&&p.job_id!==job.id;});
    const preview=await finish();
    check(preview.spoken_text.includes('Cosette'),'reader displays the actual preset preview text');
    check((await get('history')).length===historyCount,'temporary preset preview finishes without creating history');
    const form=$('#voice-form');
    form.elements.name.value='Pocket narrator';form.elements.path.value=window.__SAYIT_UI_FIXTURE__.voice_file;
    form.elements.reference_text.value='';form.elements.consent.checked=true;form.requestSubmit();
    const profile=await until(async()=>(await get('voices')).find(voice=>voice.name==='Pocket narrator'));
    check(profile.reference_kind==='audio'&&profile.model_id==='pocket-tts-int8','audio reference creates a Pocket profile');
    await until(()=>$(`[data-voice="${profile.id}"][data-voice-action=select]`));
    $(`[data-voice="${profile.id}"][data-voice-action=select]`).click();
    await until(async()=>(await get('settings')).active_voice_profile_id===profile.id);
    const cloned=await post('jobs',{text:'This recording uses the saved voice profile.',queue_policy:'append'});
    check(cloned.voice_profile_id===profile.id,'saved profile overrides the selected built-in reference');
    const clonedPlayback=await finish(cloned.id);
    check(clonedPlayback.spoken_text===cloned.text,'new reading replaces the preview text in the reader');
    await until(()=>$('[data-preset-voice=javert][data-preset-action=select]'));
    $('[data-preset-voice=javert][data-preset-action=select]').click();
    await until(async()=>{const settings=await get('settings');const model=(await get('models')).find(model=>model.selected);return settings.active_voice_profile_id===null&&model.preset_voices.find(voice=>voice.id==='javert')?.selected;});
    check(true,'switching to a preset clears the cloned profile selection');
    await until(()=>$(`[data-voice="${profile.id}"][data-voice-action=delete]`));
    $(`[data-voice="${profile.id}"][data-voice-action=delete]`).click();
    await until(async()=>(await get('voices')).length===0);
    check(true,'unused cloned profile can be removed');
    await invoke('open_management',{section:'settings'});
    await until(()=>$('#settings-form').elements.speaking_pace.disabled);
    check(getComputedStyle($('#speech-language-field')).display==='none','models without language selection hide the unsupported control');
    await window.__TAURI__.event.emit('ui-test-result',{passed:true,checks});
  }catch(error){
    await window.__TAURI__.event.emit('ui-test-result',{passed:false,error:String(error),stack:error.stack,toast:$('#toast')?.textContent,checks,
      context:await invoke('desktop_context'),section:$('.nav-item.active')?.dataset.view,
      pocket:$('#model-list [data-model=pocket-tts-int8]')?.closest('article').textContent});
  }
})();
