(async()=>{
  const invoke=window.__TAURI__.core.invoke,get=resource=>invoke('service_get',{resource}),post=(resource,body={})=>invoke('service_post',{resource,body});
  const $=selector=>document.querySelector(selector),checks=[];
  const check=(value,message)=>{if(!value)throw Error(message);checks.push(message)};
  const until=async(predicate,timeout=120000)=>{const end=performance.now()+timeout;while(performance.now()<end){const value=await predicate();if(value)return value;await new Promise(resolve=>setTimeout(resolve,750));}throw Error('Discovery UI acceptance timed out')};
  function click(element){check(Boolean(element)&&getComputedStyle(element).display!=='none','action control is displayed');element.scrollIntoView({block:'center'});element.click()}
  async function finish(id){const p=await until(async()=>{const p=(await get('state')).playback;return p.job_id===id&&p.generation_complete&&p.duration_seconds>0?p:null});check(p.amplitudes.some(x=>x>0),'audio contains real speech amplitudes');await post('playback/seek',{seconds:p.duration_seconds-.1});await until(async()=>(await get('state')).playback.state==='completed');return p}
  try{
    const context=await invoke('desktop_context');
    async function capture(theme){if(!context.snapshots)return;applyTheme(theme);await new Promise(resolve=>setTimeout(resolve,250));let result;const unlisten=await window.__TAURI__.event.listen('ui-test-snapshot-done',event=>{if(event.payload.theme===theme)result=event.payload});await window.__TAURI__.event.emit('ui-test-snapshot',theme);await until(()=>result,10000);unlisten();check(result.ok,`native ${theme} discovery screenshot captured`)}
    await post('settings',{volume:0,compute_device:'cpu'});await invoke('open_management',{section:'voices'});
    await until(()=>!$('#voice-discovery').hidden&&$('#discovery-form').elements.language.options.length);
    check(getComputedStyle($('#voice-discovery')).display!=='none','voice discovery panel is visible for the supported backend');
    check(document.documentElement.scrollWidth<=innerWidth,'discovery panel fits the native management window');await capture('light');
    const form=$('#discovery-form');check(form.elements.candidate_count.value==='4','discovery defaults to four samples');
    check(form.elements.sample_text.getBoundingClientRect().width>=form.getBoundingClientRect().width-4,'sample text uses the full form width');
    form.elements.candidate_count.value='1';form.elements.language.value='es';form.elements.language.dispatchEvent(new Event('change',{bubbles:true}));form.elements.seed.value='417';
    check(form.elements.sample_text.value.startsWith('La luz'),'sample passage follows the selected language');
    check(form.checkValidity(),'generation fields are valid: '+[...form.elements].filter(field=>field.validity&&!field.validity.valid).map(field=>field.name+': '+field.validationMessage).join(', '));form.requestSubmit();
    await until(()=>!$('#discovery-cancel').hidden);
    check(form.elements.sample_text.disabled,'generation locks the sample settings while it is running');
    const snapshot=await until(async()=>{const s=await get('voices/discovery');if(s?.state==='failed')throw Error(s.error);return s?.state==='ready'?s:null});
    const candidate=snapshot.candidates[0];check(snapshot.completed===1&&candidate.language==='es'&&!candidate.save_error,'completed candidate has language and reusable reference audio');
    const card=await until(()=>$(`[data-candidate="${candidate.id}"]`));
    card.scrollIntoView({block:'center'});await capture('dark');
    click(card.querySelector('[data-candidate-action=preview]'));
    const preview=await finish(candidate.id);check(preview.spoken_text===candidate.sample_text,'preview reader uses the candidate sample text');
    check((await get('history')).length===0,'candidate previews do not create speech history');
    card.querySelector('[data-candidate-name]').value='Mi narrador';click(card.querySelector('[data-candidate-action=save]'));
    const profile=await until(async()=>(await get('voices')).find(voice=>voice.name==='Mi narrador'));
    check(profile.generated&&profile.language==='es'&&profile.generation_tuning.top_k===40,'saved voice preserves generated origin, language, and tuning');
    await until(()=>$(`[data-candidate="${candidate.id}"] [data-candidate-action=select]`)?.hidden===false);
    click($(`[data-candidate="${candidate.id}"] [data-candidate-action=select]`));
    await until(async()=>(await get('settings')).active_voice_profile_id===profile.id);
    await until(()=>$(`[data-voice="${profile.id}"][data-voice-action=generation]`));
    click($(`[data-voice="${profile.id}"][data-voice-action=generation]`));
    await until(()=>$('#voice-generation-dialog').open);
    const editor=$('#voice-generation-form');editor.elements.preset.value='faithful';editor.elements.preset.dispatchEvent(new Event('change',{bubbles:true}));editor.requestSubmit();
    await until(()=>!$('#voice-generation-dialog').open);
    check((await get('voices')).find(voice=>voice.id===profile.id).generation_tuning.temperature===0.45,'profile generation settings save through the desktop');
    const job=await post('jobs',{text:'Buenos días. La mañana está tranquila.',queue_policy:'append'});await finish(job.id);
    check(job.voice_profile_id===profile.id&&job.language==='es','saved generated voice reads new text with its language');
    form.requestSubmit();await until(()=>!$('#discovery-cancel').hidden);click($('#discovery-cancel'));
    await until(async()=>(await get('voices/discovery')).state==='cancelled');
    check((await get('voices/discovery')).candidates.length===1,'cancellation keeps completed candidates');
    await until(()=>!$('#discovery-clear').hidden);click($('#discovery-clear'));
    await until(()=>$('#discovery-candidates').children.length===0);
    check((await get('voices')).some(voice=>voice.id===profile.id),'clearing temporary candidates preserves saved profiles');
    await window.__TAURI__.event.emit('ui-test-result',{passed:true,checks});
  }catch(error){await window.__TAURI__.event.emit('ui-test-result',{passed:false,error:String(error),stack:error.stack,toast:$('#toast')?.textContent,checks})}
})();
