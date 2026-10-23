(async()=>{
  const invoke=window.__TAURI__.core.invoke;
  const get=resource=>invoke('service_get',{resource});
  const post=(resource,body={})=>invoke('service_post',{resource,body});
  const $=selector=>document.querySelector(selector),checks=[];
  const check=(condition,message)=>{if(!condition)throw Error(message);checks.push(message)};
  const until=async(predicate,timeout=90000)=>{
    const end=performance.now()+timeout;
    while(performance.now()<end){const value=await predicate();if(value)return value;await new Promise(resolve=>setTimeout(resolve,500));}
    throw Error('Language UI acceptance timed out');
  };
  async function choose(language){
    await invoke('open_management',{section:'settings'});
    await until(()=>!$('#speech-language-field').hidden&&!$('#speech-language').disabled);
    const select=$('#speech-language');select.value=language;
    select.dispatchEvent(new Event('change',{bubbles:true}));
    await until(async()=>(await get('models')).find(model=>model.selected)?.selected_language===language);
  }
  async function finish(job){
    await until(async()=>{
      if((await get('history')).some(item=>item.job.id===job.id))return true;
      const p=(await get('state')).playback;
      if(p.job_id===job.id&&p.generation_complete&&p.duration_seconds>0){
        check(p.amplitudes.some(value=>value>0),'selected language produces real speech');
        await post('playback/seek',{seconds:p.duration_seconds-0.1});
      }
      return false;
    });
    const item=(await get('history')).find(item=>item.job.id===job.id);
    check(item.job.state==='completed'&&item.job.language===job.language,'history retains the language used by the job');
    return item;
  }
  try{
    check(Boolean(window.__SAYIT_UI_FIXTURE__.reference_text),'test reference has an exact transcript');
    await post('settings',{volume:0,compute_device:'cpu',speaking_pace:1});
    await invoke('open_management',{section:'settings'});
    await until(()=>!$('#speech-language-field').hidden);
    check($('#speech-language').options.length===11,'Qwen offers ten named languages and automatic detection');
    check(getComputedStyle($('#speech-language-field')).display!=='none','supported language control is visible');
    await choose('es');
    check($('#speech-language').value==='es','desktop saves the selected language');
    await invoke('open_management',{section:'voices'});
    await until(()=>!$('#voice-form').hidden);
    const form=$('#voice-form');form.elements.name.value='Language acceptance';
    form.elements.path.value=window.__SAYIT_UI_FIXTURE__.voice_file;
    form.elements.reference_text.value=window.__SAYIT_UI_FIXTURE__.reference_text;
    form.elements.consent.checked=true;form.requestSubmit();
    const profile=await until(async()=>(await get('voices')).find(voice=>voice.name==='Language acceptance'));
    await until(()=>$(`[data-voice="${profile.id}"][data-voice-action=select]`));
    $(`[data-voice="${profile.id}"][data-voice-action=select]`).click();
    await until(async()=>(await get('settings')).active_voice_profile_id===profile.id);
    const spanish=await post('jobs',{text:'Buenos días. La mañana está tranquila.',queue_policy:'append'});
    check(spanish.language==='es','new reading captures the selected Spanish language');
    await choose('fr');
    check((await get('settings')).active_voice_profile_id===profile.id,'changing language preserves the selected voice profile');
    const french=await post('jobs',{text:'Bonjour. La matinée est calme.',queue_policy:'append'});
    check(french.language==='fr','next reading captures the changed French language');
    const jobs=await get('jobs');
    check(jobs.find(job=>job.id===spanish.id).language==='es','changing the default leaves the queued language intact');
    await finish(spanish);await finish(french);
    await window.__TAURI__.event.emit('ui-test-result',{passed:true,checks});
  }catch(error){
    await window.__TAURI__.event.emit('ui-test-result',{passed:false,error:String(error),stack:error.stack,toast:$('#toast')?.textContent,checks});
  }
})();
