// Actual native webview + service + verified installed pack. No audio device or
// inference is used here; real synthesis has separate packaged-runtime checks.
(async()=>{
  const invoke=window.__TAURI__.core.invoke,get=resource=>invoke('service_get',{resource});
  const $=selector=>document.querySelector(selector),checks=[];
  const check=(condition,message)=>{if(!condition)throw Error(message);checks.push(message)};
  const until=async predicate=>{const end=performance.now()+45000;while(performance.now()<end){const result=await predicate();if(result)return result;await new Promise(resolve=>setTimeout(resolve,200));}throw Error('Kokoro control acceptance timed out')};
  const model=async()=>(await get('models')).find(item=>item.selected);
  const card=id=>$(`[data-preset-voice="${id}"][data-preset-action=preview]`)?.closest('article');
  try{
    await invoke('open_management',{section:'voices'});
    await until(()=>$('[data-preset-action=preview]')&&document.querySelectorAll('[data-preset-action=preview]').length===103);
    const initial=await model();
    check(initial.id==='kokoro-int8-multi-lang-v1-1'&&initial.installed,'native voice studio uses the actual installed Kokoro pack');
    check(initial.preset_voices.length===103&&initial.preset_voices.find(voice=>voice.selected)?.id==='af_maple','fresh pack shows all 103 voices with Maple selected');
    check(initial.languages.join(',')==='en-US,en-GB,zh-CN','model reports only the three installed locale groups');
    check(getComputedStyle($('#voice-form')).display==='none'&&getComputedStyle($('#voice-file-form')).display==='none'&&getComputedStyle($('#random-voice-modes')).display==='none','unsupported cloning, voice files and random modes are hidden');
    check(initial.preset_voices.every(voice=>card(voice.id)?.querySelector('h3')?.textContent===voice.name&&card(voice.id)?.querySelector('.tag')?.textContent===voice.language),'every native voice card matches its actual catalog name and locale');
    check(document.documentElement.scrollWidth<=innerWidth,'all voice cards fit the native management window');
    for(const [id,locale] of [['af_sol','en-US'],['bf_vale','en-GB'],['zf_001','zh-CN'],['af_maple','en-US']]){
      const button=$(`[data-preset-voice="${id}"][data-preset-action=select]`);
      check(button&&!button.disabled,`${id} can be selected in the native voice studio`);
      button.scrollIntoView({block:'center'});button.click();
      await until(async()=>{const chosen=(await model()).preset_voices.find(voice=>voice.selected);return chosen?.id===id&&card(id)?.querySelector('button[disabled]')?.textContent==='In use'});
      check((await model()).preset_voices.find(voice=>voice.selected).language===locale&&card(id).querySelector('.accent')?.textContent==='Selected',`${id} selection updates the real service and native selection indicator`);
    }
    $('[data-view=settings]').click();
    await until(()=>$('#settings-view').classList.contains('active')&&$('#diagnostics').textContent.includes('Platform'));
    const pace=$('#settings-form').elements.speaking_pace;
    check(!pace.disabled&&$('#speech-language-field').hidden&&$('#speech-language').disabled,'settings expose native pace and keep the unsupported language override hidden');
    pace.value='1.25';$('#settings-form').requestSubmit();
    await until(async()=>(await get('settings')).speaking_pace===1.25);
    check((await model()).preset_voices.find(voice=>voice.selected)?.id==='af_maple','saving native speaking pace preserves the selected voice');
    $('[data-view=voices]').click();
    await until(()=>$('#voices-view').classList.contains('active')&&card('af_maple')?.querySelector('button[disabled]')?.textContent==='In use');
    check(card('af_maple').querySelector('.accent')?.textContent==='Selected','voice selection survives settings navigation and service refresh');
    check((await get('history')).length===0&&(await get('voices')).length===0&&(await get('jobs')).length===0,'control acceptance creates no speech jobs, history, or saved profiles');
    await window.__TAURI__.event.emit('ui-test-result',{passed:true,inference_tested:false,audio_device_tested:false,checks});
  }catch(error){await window.__TAURI__.event.emit('ui-test-result',{passed:false,error:String(error),stack:error.stack,toast:$('#toast')?.textContent,checks})}
})();
