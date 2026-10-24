// Real service/configuration and native webview; synthesis and audio devices
// are deliberately covered by their separate acceptance suites.
(async()=>{
  const invoke=window.__TAURI__.core.invoke,get=resource=>invoke('service_get',{resource});
  const $=selector=>document.querySelector(selector),checks=[];
  const check=(condition,message)=>{if(!condition)throw Error(message);checks.push(message)};
  const until=async predicate=>{const end=performance.now()+45000;while(performance.now()<end){const result=await predicate();if(result)return result;await new Promise(resolve=>setTimeout(resolve,200));}throw Error('Qwen control acceptance timed out')};
  const modeButton=mode=>$(`[data-voice-mode="${mode}"][data-mode-action=select]`);
  const active=async()=>(await get('models')).find(model=>model.selected);
  const change=field=>field.dispatchEvent(new Event('change',{bubbles:true}));
  try{
    await invoke('open_management',{section:'voices'});
    await until(()=>modeButton('automatic_stable')?.disabled);
    const model=await active();
    check(model.id==='qwen3-tts-06b-base'&&model.inference_backend==='onnx'&&model.installed&&!model.requires_voice_profile,'native Base controls use a real installed pack without requiring a recorded profile');
    check(!$('#random-voice-modes').hidden&&!$('#voice-discovery').hidden,'reading modes and discovery appear in the native voice studio');
    check($('#random-voice-modes').getBoundingClientRect().bottom<=$('#voice-discovery').getBoundingClientRect().top,'reading modes precede discovery and recording');
    check(!$('#voice-form').hidden&&$('#voice-form').elements.reference_text.required&&$('#voice-form').elements.consent.required&&$('#voice-form').dataset.maximumSeconds==='45','Qwen recording form requires its transcript and speaker permission');
    check($('#voice-file-form').hidden&&document.querySelectorAll('[data-preset-action]').length===0,'Base hides unsupported imported voice files and fixed presets');
    check(document.documentElement.scrollWidth<=innerWidth,'Qwen voice controls fit the native management window');
    modeButton('random_per_paragraph').click();
    await until(async()=>(await get('settings')).voice_mode_by_model[model.id]==='random_per_paragraph'&&modeButton('random_per_paragraph')?.disabled);
    check((await get('settings')).active_voice_profile_id===null,'paragraph mode updates the real service without creating a profile');
    modeButton('automatic_stable').click();
    await until(async()=>(await get('settings')).voice_mode_by_model[model.id]==='automatic_stable'&&modeButton('automatic_stable')?.disabled);
    check(true,'stable reading mode can be selected again');
    const form=$('#discovery-form');
    check([...form.elements.language.options].map(option=>option.value).join(',')===model.languages.join(','),'discovery offers exactly the installed model languages');
    form.elements.preset.value='faithful';change(form.elements.preset);
    check(Number(form.elements.temperature.value)===.45&&Number(form.elements.top_k.value)===20,'Faithful style updates the numeric generation controls');
    form.elements.character.value='1';form.elements.character.dispatchEvent(new Event('input',{bubbles:true}));
    check(form.elements.preset.value==='expressive'&&Number(form.elements.temperature.value)===.85,'voice-character slider updates its style and numeric controls');
    form.elements.seed.value='12345';form.elements.randomize_tuning.checked=true;change(form.elements.randomize_tuning);
    check(form.elements.character.disabled&&form.elements.temperature.disabled&&!form.elements.seed.disabled&&form.elements.seed.value==='12345','random tuning disables manual style controls and preserves the editable seed');
    form.elements.randomize_tuning.checked=false;change(form.elements.randomize_tuning);
    check(!form.elements.character.disabled&&!form.elements.temperature.disabled,'manual tuning becomes available again');
    for(const [language,prefix] of [['es','La luz'],['ja','朝の光']]){
      form.elements.language.value=language;change(form.elements.language);
      check(form.elements.sample_text.value.startsWith(prefix),`${language} discovery uses a matching default passage`);
    }
    $('[data-view=settings]').click();
    await until(()=>$('#settings-view').classList.contains('active')&&!$('#speech-language').disabled&&$('#diagnostics').textContent.includes('Platform'));
    check(!$('#speech-language-field').hidden&&$('#settings-form').elements.speaking_pace.disabled,'Qwen settings expose language and hide unsupported native pace');
    check([...$('#speech-language').options].map(option=>option.value).join(',')===model.languages.join(','),'settings offer exactly the installed model languages');
    for(const language of [...model.languages.filter(value=>value!=='auto'),'auto']){
      const field=$('#speech-language');field.value=language;change(field);
      await until(async()=>(await active()).selected_language===language&&!$('#speech-language').disabled&&$('#speech-language').dataset.saved===language);
      check($('#speech-language').value===language,`${language} selection persists through the real service`);
    }
    $('[data-view=voices]').click();await until(()=>$('#voices-view').classList.contains('active')&&modeButton('automatic_stable')?.disabled);
    check((await get('settings')).voice_mode_by_model[model.id]==='automatic_stable','reading mode survives settings navigation and service refresh');
    check((await get('jobs')).length===0&&(await get('history')).length===0&&(await get('voices')).length===0,'control acceptance creates no speech jobs, history, or saved profiles');
    await window.__TAURI__.event.emit('ui-test-result',{passed:true,inference_tested:false,audio_device_tested:false,checks});
  }catch(error){await window.__TAURI__.event.emit('ui-test-result',{passed:false,error:String(error),stack:error.stack,toast:$('#toast')?.textContent,checks})}
})();
