(async()=>{
  const invoke=window.__TAURI__.core.invoke,get=resource=>invoke('service_get',{resource}),post=(resource,body={})=>invoke('service_post',{resource,body});
  const $=selector=>document.querySelector(selector),checks=[];
  const check=(value,message)=>{if(!value)throw Error(message);checks.push(message)};
  const until=async(predicate,timeout=180000)=>{const end=performance.now()+timeout;while(performance.now()<end){const value=await predicate();if(value)return value;await new Promise(resolve=>setTimeout(resolve,2000));}throw Error('Random mode management acceptance timed out')};
  const modeButton=(mode,action)=>$(`[data-voice-mode="${mode}"][data-mode-action="${action}"]`);
  try{
    await post('settings',{volume:0,compute_device:'cpu'});await invoke('open_management',{section:'voices'});
    await until(()=>modeButton('automatic_stable','select')?.disabled);
    check(getComputedStyle($('#random-voice-modes')).display!=='none','random reading modes are offered without creating a profile');
    check($('#random-voice-modes').getBoundingClientRect().bottom<=$('#voice-discovery').getBoundingClientRect().top,'reading modes appear before discovery and recording forms');
    check(document.documentElement.scrollWidth<=innerWidth,'mode cards fit the native management window');
    modeButton('random_per_paragraph','select').click();
    const model=(await get('models')).find(model=>model.selected);
    await until(async()=>(await get('settings')).voice_mode_by_model[model.id]==='random_per_paragraph');
    await until(()=>modeButton('random_per_paragraph','select')?.disabled);
    check((await get('jobs')).length===0,'selecting a mode in the studio does not start a reading');
    const previousCard=modeButton('automatic_stable','preview');
    await post(`models/${model.id}/language`,{language:'es'});
    // Wait for the mode cards to reflect the new model metadata before clicking.
    await until(()=>modeButton('automatic_stable','preview')!==previousCard);
    modeButton('automatic_stable','preview').click();
    const job=await until(async()=>(await get('jobs')).find(item=>item.voice_mode==='automatic_stable'));
    check(job.language==='es'&&job.text==='La luz de la mañana entra por la ventana.','preview uses the selected language and its matching sample passage');
    check((await get('settings')).voice_mode_by_model[model.id]==='random_per_paragraph','previewing a mode preserves the selected reading mode');
    const playback=await until(async()=>{const p=(await get('state')).playback;if(p.job_id!==job.id){const current=(await get('jobs')).find(item=>item.id===job.id);if(current?.state==='failed')throw Error(current.error)}return p.job_id===job.id&&p.generation_complete&&p.duration_seconds>0?p:null});
    check(playback.amplitudes.some(value=>value>0)&&playback.spoken_text===job.text,'mode preview produces real speech and shows only the requested passage');
    await post('playback/seek',{seconds:playback.duration_seconds-.1});await until(async()=>(await get('state')).playback.state==='completed');
    const fixture=window.__SAYIT_UI_FIXTURE__;
    const profile=await post('voices',{name:'Saved studio narrator',reference_audio_path:fixture.voice_file,reference_text:fixture.reference_text,speaker_permission_confirmed:true});
    const saved=await until(()=>$(`[data-voice="${profile.id}"][data-voice-action="select"]`));saved.click();
    await until(async()=>(await get('settings')).active_voice_profile_id===profile.id);
    await until(()=>!modeButton('random_per_paragraph','select')?.disabled);
    check(!(await get('settings')).voice_mode_by_model[model.id],'selecting a saved voice clears the random-mode preference');
    modeButton('automatic_stable','select').click();
    await until(async()=>(await get('settings')).active_voice_profile_id===null);
    await until(()=>modeButton('automatic_stable','select')?.disabled);
    check((await get('voices')).some(voice=>voice.id===profile.id),'returning to random reading preserves the saved profile');
    await window.__TAURI__.event.emit('ui-test-result',{passed:true,checks});
  }catch(error){await window.__TAURI__.event.emit('ui-test-result',{passed:false,error:String(error),stack:error.stack,toast:$('#toast')?.textContent,checks})}
})();
