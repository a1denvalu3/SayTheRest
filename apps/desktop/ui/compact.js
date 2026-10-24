const invoke = window.__TAURI__.core.invoke;
const $ = selector => document.querySelector(selector);
const get = resource => invoke('service_get',{resource});
const post = (resource,body = null) => invoke('service_post',{resource,body});
let visibilityChanged=false;
let visible = false, snapshot = null, models = [], settings = {}, voices = [], pending = null, refreshing = false;
let refreshAgain = null;
const resources = new Set();
function toast(message) { $('#toast').textContent = message; $('#toast').classList.add('show'); setTimeout(()=>$('#toast').classList.remove('show'),3500); }
const player = new window.SayItPlaybackView({post,notify:toast,changed:()=>refresh(false)});
player.setVisible(false);

function renderModels() {
  const options = models.filter(model=>model.installed).map(model=>{
    const option = document.createElement('option'); option.value=model.id;
    option.textContent = model.name;
    option.selected = model.selected; return option;
  });
  if (!options.length) { const option=document.createElement('option'); option.value='';option.textContent='Download a model…';options.push(option); }
  $('#player-model').replaceChildren(...options);
  renderVoiceChoices();
}
function renderVoiceChoices(){
  const model=models.find(model=>model.selected),options=[];
  const add=(value,name)=>{const option=document.createElement('option');option.value=value;option.textContent=name;options.push(option)};
  for(const preset of model?.preset_voices||[])add(`preset:${preset.id}`,preset.name);
  for(const profile of voices)add(`profile:${profile.id}`,profile.name);
  if(model?.capabilities.random_voice_sampling){add('mode:automatic_stable','Random voice');add('mode:random_per_paragraph','Random voice every paragraph')}
  if(!options.length&&model&&!model.requires_voice_profile)add('default:configured','Configured voice');
  if(!options.length)add('',model?'Create a voice profile…':'Choose a model first');
  add('manage','Manage voices…');
  const select=$('#player-voice');select.replaceChildren(...options);
  const preset=model?.preset_voices?.find(voice=>voice.selected);
  select.value=settings.active_voice_profile_id?`profile:${settings.active_voice_profile_id}`:model?.capabilities.random_voice_sampling?`mode:${settings.voice_mode_by_model?.[model.id]||'automatic_stable'}`:preset?`preset:${preset.id}`:'default:configured';
  if(!select.value)select.value='';
}
function renderHistory(history) {
  const container = $('#recent-history');container.replaceChildren();
  for (const item of history.filter(item=>item.audio_path&&item.job.state==='completed').slice(0,5)) {
    const button=document.createElement('button');button.className='recent-row';
    const text=document.createElement('span');text.textContent=item.job.text.split('\n')[0];
    const date=document.createElement('small');date.textContent=new Date(item.job.created_at).toLocaleDateString(undefined,{month:'short',day:'numeric'});
    button.append(text,date);button.onclick=async()=>{try{await post(`history/${item.job.id}/replay`);await refresh(false)}catch(error){toast(String(error))}};
    container.append(button);
  }
  if (!container.children.length) {const empty=document.createElement('p');empty.className='empty-history';empty.textContent='Your recent recordings will appear here.';container.append(empty);}
}
async function refresh(details = true) {
  if (!visible) return;
  if (refreshing) { refreshAgain=Boolean(refreshAgain||details); return; }
  refreshing=true;
  try {
    snapshot=await get('state');
    player.render(snapshot.playback,snapshot.queue_depth);
    const present=Boolean(snapshot.playback.job_id);
    document.body.classList.toggle('has-recording',present);$('#idle-actions').hidden=present;
    if (details || !models.length) {
      [models,settings,voices]=await Promise.all([get('models'),get('settings'),get('voices')]);renderModels();
    }
    if (!present) renderHistory(await get('history'));
  } catch(error) { $('#playback-status').textContent='Service unavailable';toast(String(error)); }
  finally { refreshing=false;if(refreshAgain!==null){const details=refreshAgain;refreshAgain=null;refresh(details)} }
}
$('#close-player').onclick=()=>invoke('close_player');
$('#open-settings').onclick=()=>invoke('open_management',{section:'settings'});
$('#all-history').onclick=()=>invoke('open_management',{section:'history'});
$('#quit-player').onclick=()=>invoke('quit_desktop');
$('#read-clipboard').onclick=()=>invoke('read_clipboard');
$('#player-model').onchange=async event=>{
  const id=event.target.value;if(!id){await invoke('open_management',{section:'models'});return;}
  try {
    const previous=snapshot?.playback;
    await post(`models/${id}/select`);
    settings=await get('settings');
    const chosen=models.find(model=>model.id===id);
    if ((chosen?.requires_voice_profile??(chosen?.capabilities.voice_cloning&&!chosen?.capabilities.preset_voices))&&!settings.active_voice_profile_id) {
      await invoke('open_management',{section:'voices'});return;
    }
    if (previous?.spoken_text) await post('jobs',{text:previous.spoken_text,source:'desktop',queue_policy:'interrupt',confirmed_long_text:true});
    await refresh(true);
  } catch(error) {toast(String(error));await refresh(true);}
};
$('#player-voice').onchange=async event=>{
  const choice=event.target.value,model=models.find(model=>model.selected),previous=snapshot?.playback;
  try{
    if(!choice||choice==='manage'){await invoke('open_management',{section:'voices'});return}
    if(!model)return;
    const split=choice.indexOf(':'),kind=choice.slice(0,split),id=choice.slice(split+1);
    if(kind==='mode')await post(`models/${model.id}/voice-mode`,{mode:id});
    else if(kind==='profile')await post(`voices/${id}/select`);
    else if(kind==='default')await post('voices/default/select');
    else await post(`models/${model.id}/voices/${id}/select`);
    if(previous?.spoken_text)await post('jobs',{model_id:model.id,text:previous.spoken_text,source:'desktop',queue_policy:'interrupt',confirmed_long_text:true});
  }catch(error){toast(String(error))}
  finally{await refresh(true)}
};
document.addEventListener('keydown',event=>{if(event.key==='Escape')invoke('close_player')});
const visibilityReady=window.__TAURI__.event.listen('window-visibility',event=>{
  if(event.payload.label!=='player')return;
  visibilityChanged=true;visible=event.payload.visible;player.setVisible(visible);if(visible)refresh(true);
});
window.__TAURI__.event.listen('service-event',event=>{
  resources.add(event.payload?.resource);
  if(pending||!visible)return;
  pending=setTimeout(()=>{pending=null;const details=['models','settings','voices'].some(resource=>resources.has(resource));resources.clear();refresh(details)},100);
});
window.__TAURI__.event.listen('capture-status',event=>{if(!event.payload.ok)toast(event.payload.error||'Text could not be captured');refresh(false)});
visibilityReady.then(()=>invoke('window_visible')).then(value=>{if(visibilityChanged)return;visible=value;player.setVisible(value);if(value)refresh(true)});
setInterval(()=>refresh(false),15000);
