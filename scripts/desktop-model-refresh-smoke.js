(async()=>{
  const invoke=window.__TAURI__.core.invoke;
  const get=resource=>invoke('service_get',{resource});
  const $=selector=>document.querySelector(selector);
  const until=async(predicate,timeout=45000)=>{
    const end=performance.now()+timeout;
    while(performance.now()<end){if(await predicate())return;await new Promise(resolve=>setTimeout(resolve,500));}
    throw Error('Model catalog did not refresh after an external installation');
  };
  try{
    await invoke('open_management',{section:'models'});
    await until(()=>$('[data-model=pocket-tts-int8][data-action=install]'));
    // Equivalent to installing through the CLI/API while the model pane is open.
    // The desktop has no pending download action of its own to poll.
    await invoke('service_post',{resource:'models/pocket-tts-int8/install',body:{}});
    await until(async()=>{
      const model=(await get('models')).find(model=>model.id==='pocket-tts-int8');
      if(model.download?.state==='failed')throw Error(model.download.error);
      return model.installed&&model.download?.state==='installed';
    });
    await until(()=>$('[data-model=pocket-tts-int8][data-action=select]'),25000);
    await window.__TAURI__.event.emit('ui-test-result',{passed:true,checks:['visible catalog reflects an external installation without reopening the pane']});
  }catch(error){
    await window.__TAURI__.event.emit('ui-test-result',{passed:false,error:String(error),context:await invoke('desktop_context'),section:$('.nav-item.active')?.dataset.view});
  }
})();
