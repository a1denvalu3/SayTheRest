(async()=>{
  const invoke=window.__TAURI__.core.invoke;
  const get=resource=>invoke('service_get',{resource});
  const post=(resource,body={})=>invoke('service_post',{resource,body});
  const $=selector=>document.querySelector(selector);
  const checks=[];
  const visibility=[];
  await window.__TAURI__.event.listen('window-visibility',event=>{if(event.payload.label==='player')visibility.push({seconds:performance.now()/1000,visible:event.payload.visible})});
  const check=(condition,message)=>{if(!condition)throw Error(message);checks.push(message)};
  const until=async(predicate,timeout=45000)=>{
    const end=performance.now()+timeout;
    while(performance.now()<end){const value=await predicate();if(value)return value;await new Promise(resolve=>setTimeout(resolve,100));}
    throw Error('UI acceptance timed out');
  };
  try {
    const context=await invoke('desktop_context');
    async function capture(theme) {
      if(!context.snapshots)return;
      if(theme!=='idle')applyTheme(theme);
      await new Promise(resolve=>setTimeout(resolve,250));
      let result=null;
      const unlisten=await window.__TAURI__.event.listen('ui-test-snapshot-done',event=>{if(event.payload.theme===theme)result=event.payload});
      await window.__TAURI__.event.emit('ui-test-snapshot',theme);
      await until(()=>result,10000);unlisten();check(result.ok,`native ${theme} screenshot captured`);
    }
    await invoke('open_player');
    await until(()=>!$('#idle-actions').hidden&&$('#player-model').value);
    await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
    check(innerWidth===360,`native player width is 360 pixels (observed ${innerWidth})`);
    check(document.documentElement.scrollWidth<=innerWidth,'compact layout has no horizontal overflow');
    check($('#play-toggle').disabled,'idle transport does not pretend audio is playable');
    await post('settings',{volume:0,compute_device:'cpu'});
    const text='The quiet morning brings a clear new sentence. Streaming speech begins before the final sentence is generated. Every word can be followed and revisited. '.repeat(12).trim();
    const job=await post('jobs',{text,queue_policy:'append'});
    await until(()=>document.querySelectorAll('.reader-word').length>100&&!$('#play-toggle').disabled);
    $('#play-toggle').click();
    await until(()=>$('#playback-status').textContent==='Paused');
    const position=(await get('state')).playback.position_seconds;
    await new Promise(resolve=>setTimeout(resolve,250));
    check(Math.abs((await get('state')).playback.position_seconds-position)<0.03,'native pause holds playback position');
    const complete=await until(async()=>{const p=(await get('state')).playback;return p.generation_complete?p:null;});
    check(complete.spoken_chunks.length>1,'reader uses measured synthesis block timing');
    check(complete.amplitudes.some(value=>value>0),'waveform contains real speech amplitudes');
    const words=Array.from(document.querySelectorAll('.reader-word'));
    const target=words[Math.floor(words.length*0.65)];
    await until(()=>target.getAttribute('aria-disabled')==='false');
    target.click();
    const moved=await until(async()=>{const p=(await get('state')).playback;return p.position_seconds>complete.duration_seconds*0.4?p:null;});
    check(moved.state==='paused',`word seeking preserves pause (before ${complete.state} at ${complete.position_seconds.toFixed(2)}, after ${moved.state} at ${moved.position_seconds.toFixed(2)})`);
    await until(()=>document.querySelector('.reader-word.current')!==null);
    const viewport=$('#reader-viewport');
    viewport.dispatchEvent(new WheelEvent('wheel',{deltaY:-100,bubbles:true}));viewport.scrollTop=0;
    check(!$('#follow-reader').hidden,'manual scrolling exposes resume-follow control');
    $('#follow-reader').click();
    check($('#follow-reader').hidden&&viewport.scrollTop>0,'resume-follow returns to the spoken word');
    target.focus();target.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
    check(Number(document.activeElement.dataset.word)===Number(target.dataset.word)+1,'reader supports keyboard word navigation');
    $('#rate-toggle').click();check(!$('#rate-pane').hidden,'speed adjustment opens in the compact player');
    $('#rate').value='1.5';$('#rate').dispatchEvent(new Event('change',{bubbles:true}));
    await until(async()=>(await get('settings')).playback_rate===1.5);
    check(true,'speed adjustment reaches the real service');
    $('#volume-toggle').click();$('#volume').value='0.35';$('#volume').dispatchEvent(new Event('input',{bubbles:true}));
    check($('#volume-value').textContent==='35%','volume adjustment previews its value');
    $('#volume').value='0';$('#volume').dispatchEvent(new Event('change',{bubbles:true}));
    $('#rate-pane').hidden=true;$('#volume-pane').hidden=true;
    await capture('light');await capture('dark');
    await post('playback/seek',{seconds:complete.duration_seconds-0.25});
    await until(()=>$('#play-toggle').dataset.playing==='false');$('#play-toggle').click();
    await until(()=>$('#playback-status').textContent==='Finished');
    check(!$('#reader-section').hidden,'finished recording remains available in the reader');
    $('#play-toggle').click();await until(()=>$('#play-toggle').dataset.playing==='true');
    await post('playback/seek',{seconds:complete.duration_seconds-0.25});
    await until(()=>$('#playback-status').textContent==='Finished');
    check((await get('history')).filter(item=>item.job.id===job.id).length===1,'replaying a retained recording does not duplicate history');
    $('[data-control=stop]').click();
    await until(()=>!$('#idle-actions').hidden&&document.querySelectorAll('.recent-row').length>0);
    check(true,'clearing returns to clipboard and recent history');
    await capture('idle');
    $('#open-settings').click();await until(async()=>!(await invoke('window_visible')));
    check(true,'settings opens the management window and hides the player');
    await window.__TAURI__.event.emit('ui-test-result',{passed:true,checks});
  } catch(error) {
    const playback=(await get('state')).playback;
    delete playback.amplitudes;delete playback.spoken_text;delete playback.spoken_chunks;
    await window.__TAURI__.event.emit('ui-test-result',{passed:false,error:String(error),stack:error.stack,viewport:{width:innerWidth,height:innerHeight,scale:devicePixelRatio},playback,visible:await invoke('window_visible'),visibility,reader:{status:$('#playback-status').textContent,words:document.querySelectorAll('.reader-word').length,available:document.querySelectorAll('.reader-word[aria-disabled=false]').length},checks});
  }
})();
