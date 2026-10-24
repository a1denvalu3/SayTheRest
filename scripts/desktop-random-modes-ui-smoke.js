(async()=>{
  const invoke=window.__TAURI__.core.invoke,get=resource=>invoke('service_get',{resource}),post=(resource,body={})=>invoke('service_post',{resource,body});
  const $=selector=>document.querySelector(selector),checks=[];
  const check=(value,message)=>{if(!value)throw Error(message);checks.push(message)};
  const until=async(predicate,timeout=180000)=>{const end=performance.now()+timeout;while(performance.now()<end){const value=await predicate();if(value)return value;await new Promise(resolve=>setTimeout(resolve,2000));}throw Error('Random reading UI acceptance timed out')};
  async function choose(value){await until(()=>$(`option[value="${value}"]`));const select=$('#player-voice');select.value=value;select.dispatchEvent(new Event('change',{bubbles:true}))}
  async function finish(id){const p=await until(async()=>{const p=(await get('state')).playback;if(p.job_id!==id){const job=(await get('jobs')).find(job=>job.id===id);if(job?.state==='failed')throw Error(job.error)}return p.job_id===id&&p.generation_complete&&p.duration_seconds>0?p:null});check(p.amplitudes.some(value=>value>0),'selected voice produces real speech amplitudes');await post('playback/seek',{seconds:p.duration_seconds-.1});await until(async()=>(await get('state')).playback.state==='completed');return p}
  try{
    await post('settings',{volume:0,compute_device:'cpu'});await invoke('open_player');
    await until(()=>$('#player-voice').value==='mode:automatic_stable');
    check(innerWidth===360&&document.documentElement.scrollWidth<=innerWidth,'voice picker fits the 360-pixel native player');
    check($('#player-voice').getBoundingClientRect().bottom<=innerHeight,'voice choices remain visible in the compact footer');
    const model=(await get('models')).find(model=>model.selected);
    check(model.capabilities.random_voice_sampling&&!model.requires_voice_profile,'native Qwen is ready without a saved profile');
    await choose('mode:random_per_paragraph');
    await until(async()=>(await get('settings')).voice_mode_by_model[model.id]==='random_per_paragraph');
    check((await get('jobs')).length===0,'choosing an idle reading mode does not start speech');
    await choose('mode:automatic_stable');await until(async()=>(await get('settings')).voice_mode_by_model[model.id]==='automatic_stable');
    const fixture=window.__SAYIT_UI_FIXTURE__;
    check(Boolean(fixture.voice_file&&fixture.reference_text),'test profile has a matching local synthetic reference');
    const profile=await post('voices',{name:'Saved test narrator',reference_audio_path:fixture.voice_file,reference_text:fixture.reference_text,speaker_permission_confirmed:true});
    await choose(`profile:${profile.id}`);await until(async()=>(await get('settings')).active_voice_profile_id===profile.id);
    check($('#player-voice').value===`profile:${profile.id}`,'a newly created profile appears and can be selected without reopening');
    await choose('mode:automatic_stable');await until(async()=>(await get('settings')).active_voice_profile_id===null);
    const job=await post('jobs',{text:'The morning is quiet and the words are clear.',queue_policy:'append'});
    check(job.voice_mode==='automatic_stable'&&job.voice_profile_id===null,'random mode overrides the previously selected saved profile');
    const first=await finish(job.id);check(first.spoken_text===job.text,'reader shows the requested text during automatic voice reading');
    await choose(`profile:${profile.id}`);
    const replay=await until(async()=>(await get('jobs')).find(item=>item.id!==job.id&&item.voice_profile_id===profile.id));
    check((await get('jobs')).find(item=>item.id===job.id).state==='completed','changing a finished voice preserves its completed job record');
    check(replay.voice_mode==='standard'&&replay.text===job.text,'changing the retained reading to a saved voice queues its exact text');await finish(replay.id);
    await choose('mode:random_per_paragraph');
    const random=await until(async()=>(await get('jobs')).find(item=>item.voice_mode==='random_per_paragraph'));
    check(random.voice_profile_id===null&&random.text===job.text,'compact mode change re-reads the retained text with the requested random mode');
    await until(async()=>(await get('jobs')).find(item=>item.id===random.id)?.state==='synthesizing');await post('playback/stop',{});
    await until(async()=>(await get('jobs')).find(item=>item.id===random.id)?.state==='cancelled');
    check(!(await get('history')).some(item=>item.job.id===random.id),'cancelled voice preparation publishes no recording');
    await choose('manage');await until(async()=>!(await invoke('window_visible')));
    check(true,'compact voice picker opens voice management');
    await window.__TAURI__.event.emit('ui-test-result',{passed:true,checks});
  }catch(error){await window.__TAURI__.event.emit('ui-test-result',{passed:false,error:String(error),stack:error.stack,toast:$('#toast')?.textContent,checks})}
})();
