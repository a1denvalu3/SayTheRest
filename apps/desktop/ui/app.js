const invoke = window.__TAURI__.core.invoke;
const saveFile = window.__TAURI__.dialog.save;
const $ = (selector) => document.querySelector(selector);
const views = {player:'Listen without leaving.',models:'Choose the engine.',voices:'A voice, held locally.',history:'What you heard.',settings:'Make it yours.'};
let snapshot = null;
let historyItems = [];
let recordingTimer = null;
let recordingStopping = false;
let preparedRecordingPath = null;
let recordingTranscriptAutofilled = false;
let voiceProfiles = [];
const shortVoicePassage = 'The morning light filled the room. I opened the window, took a deep breath, and began reading in my usual voice.';
const longVoicePassage = 'On a clear morning, I stepped outside and listened to the neighborhood waking up. A bicycle rolled past, birds called from the trees, and someone laughed across the street. The air felt cool, but the sunlight was warm. I paused, took a slow breath, and wondered what the day might bring. Would it be quiet and familiar, or full of small surprises? Whatever happened, I wanted to speak clearly, stay curious, and enjoy the simple details that make an ordinary day memorable.';
let windowVisible = false;
let visibilityChanged = false;
let playbackView = null;
let availableUpdate = null;
let serviceEventTimer = null;
let modelDownloadActive = false;
let modelRefreshInFlight = false;
let modelsLoaded = false;
const pendingServiceResources = new Set();

document.querySelectorAll('.nav-item').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.nav-item,.view').forEach(item => item.classList.remove('active'));
  button.classList.add('active');
  $(`#${button.dataset.view}-view`).classList.add('active');
  $('#view-title').textContent = views[button.dataset.view];
  $('#eyebrow').textContent = button.textContent.replace(/^\d+/, '').trim().toUpperCase();
  refresh(button.dataset.view);
}));

async function get(resource){ return invoke('service_get',{resource}); }
async function post(resource,body=null){ return invoke('service_post',{resource,body}); }
async function del(resource){ return invoke('service_delete',{resource}); }
function toast(message){const el=$('#toast');el.textContent=message;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),2600)}
function time(value=0){const seconds=Math.max(0,Math.floor(value));return `${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}`}

async function refresh(section='player',loadDetails=true){
  if(!windowVisible)return;
  try{
    snapshot=await get('state');
    $('#service-pill').classList.remove('offline');$('#service-pill').innerHTML='<i></i>Service ready';
    await renderPlayer(snapshot);
    if(section==='models'&&(loadDetails||!modelsLoaded)) renderModels(await get('models'));
    if(loadDetails&&section==='voices'){const [voices,settings,models]=await Promise.all([get('voices'),get('settings'),get('models')]);renderVoices(voices,settings,models)}
    if(loadDetails&&section==='history'){historyItems=await get('history');renderHistory(historyItems)}
    if(loadDetails&&section==='settings') await renderSettings(await get('settings'));
  }catch(error){$('#service-pill').classList.add('offline');$('#service-pill').innerHTML='<i></i>Service offline';console.error(error)}
}

function renderPlayer(state){ playbackView?.render(state.playback,state.queue_depth); }

function inferenceDeviceLabel(device){if(!device?.device)return 'Not loaded yet';return [device.device.toUpperCase(),device.backend==='qwen3-tts-onnx'?'Native ONNX':'',device.quantization?.toUpperCase()].filter(Boolean).join(' · ')}
function renderModels(models){modelsLoaded=true;modelDownloadActive=models.some(model=>model.download&&['downloading','verifying','installing'].includes(model.download.state));$('#model-list').innerHTML=models.map(model=>{const d=model.download;const busy=d&&['downloading','verifying','installing'].includes(d.state);const progress=d?Math.round(d.downloaded_bytes/Math.max(1,d.total_bytes)*100):0;const primary=model.selected?'<button class="action secondary" disabled>In use</button>':`<button class="action ${model.installed?'secondary':''}" data-model="${escapeHtml(model.id)}" data-action="${model.installed?'select':'install'}">${model.installed?'Use this model':'Download'}</button>`;const benchmark=model.benchmark?`<div class="benchmark"><strong>${model.benchmark.mean_rtf.toFixed(2)}× RTF</strong><span>cold ${Math.round(model.benchmark.cold_start_ms)} ms · p95 ${Math.round(model.benchmark.p95_ms)} ms</span></div>`:'';const installedActions=model.installed?`${model.update_available?`<button class="action secondary" data-model="${escapeHtml(model.id)}" data-action="update">Update</button>`:''}<button class="action secondary" data-model="${escapeHtml(model.id)}" data-action="benchmark">Benchmark</button>${model.resident?`<button class="action secondary" data-model="${escapeHtml(model.id)}" data-action="unload">Unload</button>`:''}${model.selected?'':`<button class="action secondary danger" data-model="${escapeHtml(model.id)}" data-action="remove">Remove</button>`}`:'';return `<article class="data-card"><header><div><h3>${escapeHtml(model.name)}</h3><div class="tags"><span class="tag">${escapeHtml(model.languages.join(', '))}</span>${model.capabilities.voice_cloning?'<span class="tag accent">Voice cloning</span>':''}${model.inference_backend==='onnx'?`<span class="tag">Native ONNX · ${escapeHtml((model.weight_precision||'').toUpperCase())}</span>`:''}${model.selected?'<span class="tag accent">Selected</span>':''}${model.resident?'<span class="tag accent">Loaded</span>':''}${model.inference_device?`<span class="tag">${escapeHtml(inferenceDeviceLabel(model.inference_device))}</span>`:''}${model.update_available?'<span class="tag accent">Update available</span>':''}${model.recommended?'<span class="tag accent">Fastest measured</span>':''}</div></div><span>${Math.round(model.size_bytes/1048576)} MB</span></header><p>${escapeHtml(model.quality_note)}</p><p><small>${escapeHtml(model.speed_note)} · ${escapeHtml(model.license)} license</small></p>${model.inference_device?.fallback_reason?`<p>${escapeHtml(model.inference_device.fallback_reason)}</p>`:model.inference_device?.provider==='cuda'?'<p>CUDA execution provider · unsupported operations may run on the CPU.</p>':''}${benchmark}${busy?`<div class="progress"><div style="width:${progress}%"></div></div><small>${escapeHtml(d.state)} · ${progress}%</small><div class="card-actions"><button class="action secondary danger" data-model="${escapeHtml(model.id)}" data-action="cancel">Cancel</button></div>`:`<div class="card-actions">${primary}${installedActions}${model.id==='qwen3-tts-06b-base'?`<button class="action secondary" data-model="${escapeHtml(model.id)}" data-action="${model.inference_backend==='onnx'?'backend-pytorch':'backend-onnx'}">${model.inference_backend==='onnx'?'Use PyTorch':'Try native INT4'}</button>`:''}</div>`}</article>`}).join('');document.querySelectorAll('[data-model]').forEach(button=>button.onclick=()=>modelAction(button))}
async function pollModelProgress(){if(!windowVisible||!modelDownloadActive||modelRefreshInFlight||document.querySelector('.nav-item.active')?.dataset.view!=='models')return;modelRefreshInFlight=true;try{renderModels(await get('models'))}catch(error){console.error(error)}finally{modelRefreshInFlight=false}}
async function modelAction(button){const {model,action}=button.dataset;try{button.disabled=true;if(action.startsWith('backend-'))await post(`models/${model}/backend`,{backend:action.slice(8),precision:'int4'});else if(action==='remove')await del(`models/${model}`);else await post(`models/${model}/${action}`);if(action==='select'){const [models,settings]=await Promise.all([get('models'),get('settings')]);const selected=models.find(candidate=>candidate.id===model);if((selected?.requires_voice_profile??(selected?.capabilities.voice_cloning&&!selected?.capabilities.preset_voices))&&!settings.active_voice_profile_id){toast('Model selected · choose or create its voice');document.querySelector('[data-view="voices"]').click();return}}toast(action.startsWith('backend-')?'Backend setup started':action==='install'?'Download started':action==='update'?'Model update started':action==='cancel'?'Download cancellation requested':action==='remove'?'Model removed':action==='unload'?'Model unloaded':action==='benchmark'?'Local benchmark complete':'Model selected');refresh('models')}catch(error){button.disabled=false;toast(String(error))}}

function renderVoices(voices,settings,models){
  voiceProfiles=voices;
  const selected=settings.active_voice_profile_id;
  const model=models.find(candidate=>candidate.selected);
  $('#voice-form').hidden=!model?.capabilities.voice_cloning;
  $('#voice-file-form').hidden=!model?.capabilities.voice_file_import;
  const usesTranscript=Boolean(model?.id.startsWith('qwen3-tts'));
  $('#voice-form').elements.reference_text.required=usesTranscript;
  const maximumSeconds=usesTranscript?45:15;
  $('#voice-form').dataset.maximumSeconds=String(maximumSeconds);
  $('#voice-reference-intro').textContent=model?.capabilities.voice_file_import
    ?'Choose a preset or import an exported Phonon voice file. Your voices stay on this device.'
    :model?.capabilities.voice_cloning?`Record or import a clean 3–${maximumSeconds} second sample from a speaker who gave permission.`:'Choose a built-in voice and preview it locally.';
  $('#voice-recording-script').textContent=usesTranscript?longVoicePassage:shortVoicePassage;
  $('#voice-recording-limit').textContent=`Speak naturally and stop after the last word. The microphone records up to ${maximumSeconds} seconds.`;
  $('#voice-transcript-note').textContent=usesTranscript
    ?'Required for Qwen voice cloning. Check every word against your recording. For imported audio, enter its exact transcript in the original language.'
    :'After recording, check that this text matches what you actually said. A transcript is optional for PocketTTS.';
  const builtIns=(model?.preset_voices||[]).map(voice=>{
    const active=!selected&&voice.selected;
    return `<article class="data-card"><header><h3>${escapeHtml(voice.name)}</h3><div><span class="tag">${escapeHtml(voice.language)}</span>${active?' <span class="tag accent">Selected</span>':''}</div></header><p>Built into ${escapeHtml(model.name)} · processed locally.</p>${voice.license?`<p title="${escapeHtml(voice.attribution||'')}">${escapeHtml(voice.license)}</p>`:''}<div class="card-actions"><button class="action" data-preset-model="${escapeHtml(model.id)}" data-preset-voice="${escapeHtml(voice.id)}" data-preset-action="preview">Preview</button>${active?'<button class="action secondary" disabled>In use</button>':`<button class="action secondary" data-preset-model="${escapeHtml(model.id)}" data-preset-voice="${escapeHtml(voice.id)}" data-preset-action="select">Use this voice</button>`}</div></article>`;
  }).join('');
  const noVoice=!builtIns&&!voices.length?'<p class="empty">This model needs a cloned voice. Record or import a permitted reference sample below.</p>':'';
  const clones=voices.map(voice=>{
    const embedding=voice.reference_kind==='phonon-embedding';
    const active=voice.id===selected;
    const quality=voice.quality||{};
    const issues=(quality.issues||[]).map(issue=>`<li>${escapeHtml(issue)}</li>`).join('');
    const transcriptStatus=usesTranscript?`<p>${voice.reference_text?'Transcript saved · used for voice cloning.':'Add the transcript of this recording to improve voice similarity.'}</p>`:'';
    const tuning=embedding?'':usesTranscript
      ?`<button class="action secondary" data-voice="${voice.id}" data-voice-action="transcript">${voice.reference_text?'Edit':'Add'} transcript</button>`
      :`<button class="action secondary" data-voice="${voice.id}" data-voice-action="tune" data-refinement-steps="${voice.refinement_steps||5}">Tune</button>`;
    return `<article class="data-card"><header><h3>${escapeHtml(voice.name)}</h3><div><span>${embedding?'Phonon voice file':`${voice.reference_duration_seconds.toFixed(1)} sec`}</span>${active?' <span class="tag accent">Selected</span>':''}</div></header><p>${embedding?'Imported voice file':'Local reference profile'} · created ${new Date(voice.created_at).toLocaleDateString()}${usesTranscript||embedding?'':` · refinement ${voice.refinement_steps||5}/8`}</p>${transcriptStatus}${quality.sample_rate?`<div class="voice-quality"><strong>${quality.quality_score}/100</strong><span>${Math.round(quality.sample_rate/1000)} kHz · RMS ${quality.rms_dbfs.toFixed(1)} dBFS · clipped ${quality.clipped_sample_percent.toFixed(2)}%</span></div>${issues?`<ul class="quality-issues">${issues}</ul>`:''}`:''}<div class="card-actions"><button class="action" data-voice="${voice.id}" data-voice-action="preview">Preview</button>${active?'<button class="action secondary" disabled>In use</button>':`<button class="action secondary" data-voice="${voice.id}" data-voice-action="select">Use this voice</button>`}${tuning}<button class="action secondary" data-voice="${voice.id}" data-voice-action="rename" data-voice-name="${escapeHtml(voice.name)}">Rename</button>${active?'':`<button class="action secondary danger" data-voice="${voice.id}" data-voice-action="delete">Delete</button>`}</div></article>`;
  }).join('');
  $('#voice-list').innerHTML=builtIns+noVoice+clones;
  document.querySelectorAll('[data-voice]').forEach(button=>button.onclick=()=>voiceAction(button));
  document.querySelectorAll('[data-preset-voice]').forEach(button=>button.onclick=()=>presetVoiceAction(button));
}
async function refreshVoices(){const [voices,settings,models]=await Promise.all([get('voices'),get('settings'),get('models')]);renderVoices(voices,settings,models)}
async function voiceAction(button){
  const {voice:id,voiceAction:action,voiceName,refinementSteps}=button.dataset;
  try{
    if(action==='transcript'){
      const voice=voiceProfiles.find(voice=>voice.id===id);
      const form=$('#voice-transcript-form');
      form.dataset.voiceId=id;
      form.elements.reference_text.value=voice?.reference_text||'';
      $('#voice-transcript-dialog').showModal();
      return;
    }
    if(action==='delete')await del(`voices/${id}`);
    else if(action==='rename'){const name=window.prompt('Voice name',voiceName);if(!name)return;await post(`voices/${id}/rename`,{name})}
    else if(action==='tune'){const steps=window.prompt('Refinement steps (3–8). Higher can improve quality but takes longer.',refinementSteps||'5');if(steps===null)return;await post(`voices/${id}/tune`,{refinement_steps:Number(steps)})}
    else await post(`voices/${id}/${action}`);
    toast(action==='preview'?'Voice preview queued':action==='tune'?'Voice refinement saved':'Voice profiles updated');
    await refreshVoices();
  }catch(error){toast(String(error))}
}
$('#cancel-voice-transcript').onclick=()=>$('#voice-transcript-dialog').close();
$('#voice-transcript-form').onsubmit=async event=>{
  event.preventDefault();
  const form=event.target;
  const referenceText=form.elements.reference_text.value.trim();
  if(!referenceText){toast('Enter the exact words spoken in the recording');return}
  const button=form.querySelector('[type="submit"]');
  button.disabled=true;
  try{
    await post(`voices/${form.dataset.voiceId}/transcript`,{reference_text:referenceText});
    $('#voice-transcript-dialog').close();
    toast('Transcript saved · preview the voice to hear the change');
    await refreshVoices();
  }catch(error){toast(String(error))}finally{button.disabled=false}
};
async function presetVoiceAction(button){const action=button.dataset.presetAction;try{button.disabled=true;button.textContent=action==='preview'?'Generating…':button.textContent;await post(`models/${button.dataset.presetModel}/voices/${button.dataset.presetVoice}/${action}`);toast(action==='preview'?'Voice preview playing':'Built-in voice selected');if(action==='select')await refreshVoices();else{button.disabled=false;button.textContent='Preview'}}catch(error){button.disabled=false;if(action==='preview')button.textContent='Preview';toast(String(error))}}
function renderHistory(items){const query=$('#history-search').value.trim().toLocaleLowerCase();const filtered=query?items.filter(item=>item.job.text.toLocaleLowerCase().includes(query)):items;$('#history-list').innerHTML=filtered.length?filtered.map(item=>`<article class="data-card"><header><h3>${item.pinned?'◆ ':''}${escapeHtml(item.job.text.slice(0,96))}${item.job.text.length>96?'…':''}</h3><span>${time(item.duration_seconds)}</span></header><p>${new Date(item.job.created_at).toLocaleString()} · ${escapeHtml(item.job.source)}</p><div class="card-actions"><button class="action" data-history="${item.job.id}" data-history-action="replay" ${item.audio_path?'':'disabled'}>Replay</button><button class="action secondary" data-history="${item.job.id}" data-history-action="regenerate">Regenerate</button><button class="action secondary" data-history="${item.job.id}" data-history-action="export" ${item.audio_path?'':'disabled'}>Export WAV</button><button class="action secondary" data-history="${item.job.id}" data-history-action="pin" data-pinned="${item.pinned}">${item.pinned?'Unpin':'Pin'}</button><button class="action secondary danger" data-history="${item.job.id}" data-history-action="delete">Delete</button></div></article>`).join(''):`<p class="empty">${query?'No history matches your search.':'Your listening history is empty.'}</p>`;document.querySelectorAll('[data-history]').forEach(button=>button.onclick=()=>historyAction(button))}
async function historyAction(button){const {history:id,historyAction:action,pinned}=button.dataset;try{if(action==='delete')await del(`history/${id}`);else if(action==='pin')await post(`history/${id}/pin`,{pinned:pinned!=='true'});else if(action==='export'){const item=historyItems.find(entry=>entry.job.id===id);const destination=await saveFile({defaultPath:`sayit-${id}.wav`,filters:[{name:'WAV audio',extensions:['wav']}]});if(!destination)return;await invoke('export_history',{source:item.audio_path,destination})}else await post(`history/${id}/${action}`);toast(action==='replay'?'Replaying archived audio':action==='regenerate'?'Fresh audio queued':action==='export'?'WAV exported':'History updated');historyItems=await get('history');renderHistory(historyItems)}catch(error){toast(String(error))}}
function escapeHtml(text){const el=document.createElement('span');el.textContent=text;return el.innerHTML}
function ensureStartupControl(form){if(form.elements.launch_at_login)return;const label=document.createElement('label');label.className='consent';label.innerHTML='<input name="launch_at_login" type="checkbox"><span>Start the speech service and shortcut tray when I sign in</span>';form.insertBefore(label,form.querySelector('button[type=submit]'))}
function showShortcuts(settings){const display=value=>value.split('+').map(part=>part.charAt(0).toUpperCase()+part.slice(1)).join(' + ');$('#selection-shortcut-label').textContent=display(settings.selection_shortcut);$('#clipboard-shortcut-label').textContent=display(settings.clipboard_shortcut);const form=$('#shortcut-form');ensureStartupControl(form);form.elements.selection_shortcut.value=settings.selection_shortcut;form.elements.clipboard_shortcut.value=settings.clipboard_shortcut;form.elements.launch_at_login.checked=settings.launch_at_login!==false}
function ensureCleaningControls(form){if(form.querySelector('#text-cleaning-controls'))return;const fieldset=document.createElement('fieldset');fieldset.id='text-cleaning-controls';fieldset.className='settings-fieldset';fieldset.innerHTML='<legend>Text cleaning</legend><p>Clean copied document and webpage text before it enters speech or history.</p><label class="consent"><input name="cleaning_enabled" type="checkbox"><span>Enable text cleaning</span></label><div class="cleaning-options"><label class="consent"><input name="strip_markdown" type="checkbox"><span>Strip Markdown formatting</span></label><label class="consent"><input name="strip_html" type="checkbox"><span>Strip HTML formatting</span></label><label class="consent"><input name="strip_code_blocks" type="checkbox"><span>Remove fenced code blocks</span></label><label class="consent"><input name="strip_special_characters" type="checkbox"><span>Remove invisible clipboard artifacts</span></label><label class="consent"><input name="normalize_whitespace" type="checkbox"><span>Normalize whitespace</span></label></div>';form.insertBefore(fieldset,form.querySelector('button[type=submit]'));form.elements.cleaning_enabled.onchange=()=>syncCleaningControls(form)}
function syncCleaningControls(form){const enabled=form.elements.cleaning_enabled.checked;form.querySelectorAll('.cleaning-options input').forEach(input=>input.disabled=!enabled)}
async function renderSettings(settings){const form=$('#settings-form');form.elements.compute_device.value=settings.compute_device||'auto';ensureCleaningControls(form);form.elements.speaking_pace.value=String(settings.speaking_pace);form.elements.model_timeout_minutes.value=Math.round(settings.model_unload_delay_seconds/60);form.elements.history_retention_days.value=String(settings.history_retention_days||0);form.elements.history_quota_mb.value=Math.round(settings.history_quota_bytes/1048576);form.elements.long_text_characters.value=settings.long_text_confirmation_characters;const cleaning=settings.text_cleaning||{enabled:true,strip_markdown:true,strip_html:true,strip_code_blocks:true,strip_special_characters:true,normalize_whitespace:true};for(const name of ['enabled','strip_markdown','strip_html','strip_code_blocks','strip_special_characters','normalize_whitespace'])form.elements[name==='enabled'?'cleaning_enabled':name].checked=Boolean(cleaning[name]);syncCleaningControls(form);const models=await get('models');const active=models.find(model=>model.selected);form.elements.speaking_pace.disabled=!active?.supports_native_speaking_pace;$('#speaking-pace-note').textContent=active?.supports_native_speaking_pace?'Shapes phoneme timing in newly generated audio. Existing history is unchanged.':'The selected model does not expose native speaking pace; it uses its natural pace.';showShortcuts(await invoke('desktop_settings'));const diagnostics=await invoke('desktop_diagnostics');const device=snapshot?.inference_device;$('#diagnostics').innerHTML=`<h3>Diagnostics</h3><dl><div><dt>Last inference</dt><dd>${escapeHtml(inferenceDeviceLabel(device))}${device?.dtype?` · ${escapeHtml(device.dtype)}`:''}</dd></div>${device?.fallback_reason?`<div><dt>CPU fallback</dt><dd>${escapeHtml(device.fallback_reason)}</dd></div>`:''}<div><dt>Platform</dt><dd>${escapeHtml(diagnostics.platform)} / ${escapeHtml(diagnostics.architecture)}</dd></div><div><dt>Desktop session</dt><dd>${escapeHtml(diagnostics.session)}</dd></div><div><dt>Selection access</dt><dd>${escapeHtml(diagnostics.selection_support)}</dd></div><div><dt>Launch at login</dt><dd>${diagnostics.launch_at_login_active?'Enabled':diagnostics.launch_at_login_requested?'Requested, but service is not enabled':'Disabled'}</dd></div><div><dt>Service credentials</dt><dd>${diagnostics.token_available?'Ready':'Unavailable'}</dd></div></dl>`}

function setRecordingUi(status){
  const active=status.recording;
  $('#record-start').disabled=active;
  $('#record-stop').disabled=!active;
  $('#record-cancel').disabled=!active;
  $('#voice-form').elements.path.readOnly=active;
  $('#voice-form').elements.reference_text.readOnly=active;
  $('#voice-form').querySelector('[type="submit"]').disabled=active;
  $('#recording-time').textContent=`${status.elapsed_seconds.toFixed(1)} sec`;
  $('#recording-state').textContent=active?'Recording locally… read the passage above':'Use your default microphone';
}
async function pollRecording(){try{const status=await invoke('voice_recording_status');setRecordingUi(status);if(status.recording&&status.elapsed_seconds>=status.maximum_seconds-.1)await finishRecording()}catch(error){clearInterval(recordingTimer);recordingTimer=null;toast(String(error))}}
async function finishRecording(){
  if(recordingStopping)return;
  recordingStopping=true;
  try{
    const result=await invoke('stop_voice_recording');
    clearInterval(recordingTimer);recordingTimer=null;
    preparedRecordingPath=result.path;
    $('#voice-form').elements.path.value=result.path;
    setRecordingUi({recording:false,elapsed_seconds:result.duration_seconds});
    $('#record-cancel').disabled=false;$('#record-cancel').textContent='Discard';
    const maximumSeconds=Number($('#voice-form').dataset.maximumSeconds||15);
    $('#recording-state').textContent=result.duration_seconds>=maximumSeconds-.1
      ?`${maximumSeconds}-second limit reached · make sure the transcript only includes words captured`
      :'Recording ready · check the transcript matches what you said';
    toast('Recording ready · review the transcript before saving');
  }catch(error){toast(String(error))}finally{recordingStopping=false}
}
$('#record-start').onclick=async()=>{
  try{
    if(preparedRecordingPath)await invoke('discard_voice_recording',{path:preparedRecordingPath});
    const status=await invoke('start_voice_recording',{maximumSeconds:Number($('#voice-form').dataset.maximumSeconds||15)});
    preparedRecordingPath=null;
    $('#record-cancel').textContent='Cancel';
    $('#voice-form').elements.path.value='';
    $('#voice-form').elements.reference_text.value=$('#voice-recording-script').textContent.trim();
    recordingTranscriptAutofilled=true;
    setRecordingUi(status);
    $('#voice-recording-script').scrollIntoView({block:'start',behavior:'smooth'});
    recordingTimer=setInterval(pollRecording,100);
  }catch(error){toast(String(error))}
};
$('#record-stop').onclick=finishRecording;
$('#voice-form').elements.reference_text.oninput=()=>{recordingTranscriptAutofilled=false};
$('#voice-form').elements.path.oninput=event=>{
  if(recordingTranscriptAutofilled&&event.target.value.trim()!==preparedRecordingPath){
    $('#voice-form').elements.reference_text.value='';
    recordingTranscriptAutofilled=false;
  }
};
$('#record-cancel').onclick=async()=>{
  try{
    if(preparedRecordingPath){await invoke('discard_voice_recording',{path:preparedRecordingPath});preparedRecordingPath=null}
    else await invoke('cancel_voice_recording');
    $('#voice-form').elements.path.value='';
    $('#voice-form').elements.reference_text.value='';
    recordingTranscriptAutofilled=false;
    clearInterval(recordingTimer);recordingTimer=null;
    $('#record-cancel').textContent='Cancel';
    setRecordingUi({recording:false,elapsed_seconds:0});
    toast('Recording discarded');
  }catch(error){toast(String(error))}
};
$('#voice-form').onsubmit=async event=>{
  event.preventDefault();
  const form=event.target;
  const data=new FormData(form);
  const path=String(data.get('path')).trim();
  const referenceText=String(data.get('reference_text')).trim();
  if(!path){toast('Record a sample or choose a reference WAV first');return}
  if(form.elements.reference_text.required&&!referenceText){toast('Enter the exact words spoken in the recording');return}
  const button=form.querySelector('[type="submit"]');
  button.disabled=true;
  try{
    await post('voices',{name:data.get('name'),reference_audio_path:path,reference_text:referenceText||null,speaker_permission_confirmed:data.get('consent')==='on'});
    if(preparedRecordingPath)invoke('discard_voice_recording',{path:preparedRecordingPath}).catch(console.error);
    preparedRecordingPath=null;
    recordingTranscriptAutofilled=false;
    form.reset();
    setRecordingUi({recording:false,elapsed_seconds:0});
    toast('Voice profile created · choose “Use this voice” to select it');
    await refreshVoices();
  }catch(error){toast(String(error))}finally{button.disabled=false}
};
$('#history-search').oninput=()=>renderHistory(historyItems);
$('#browse-voice-file').onclick=async()=>{
  try {
    const path=await window.__TAURI__.dialog.open({multiple:false,directory:false,filters:[{name:'Phonon voice',extensions:['safetensors']}]});
    if(path)$('#voice-file-form').elements.path.value=path;
  }catch(error){toast(String(error))}
};
$('#voice-file-form').onsubmit=async event=>{
  event.preventDefault();const form=event.target,button=form.querySelector('[type=submit]');button.disabled=true;
  try {
    await post('voices/import',{kind:'phonon-embedding',name:form.elements.name.value,path:form.elements.path.value});
    form.reset();await refreshVoices();toast('Phonon voice imported · choose “Use this voice” to select it');
  }catch(error){toast(String(error))}finally{button.disabled=false}
};
$('#settings-form').onsubmit=async event=>{event.preventDefault();const data=new FormData(event.target);try{const settings=await post('settings',{compute_device:String(data.get('compute_device')),speaking_pace:event.target.elements.speaking_pace.disabled?null:Number(data.get('speaking_pace')),model_unload_delay_seconds:Number(data.get('model_timeout_minutes'))*60,history_retention_days:Number(data.get('history_retention_days')),history_quota_bytes:Number(data.get('history_quota_mb'))*1048576,long_text_confirmation_characters:Number(data.get('long_text_characters')),text_cleaning:{enabled:data.get('cleaning_enabled')==='on',strip_markdown:data.get('strip_markdown')==='on',strip_html:data.get('strip_html')==='on',strip_code_blocks:data.get('strip_code_blocks')==='on',strip_special_characters:data.get('strip_special_characters')==='on',normalize_whitespace:data.get('normalize_whitespace')==='on'}});renderSettings(settings);toast('Settings saved')}catch(error){toast(String(error))}};
async function checkForUpdates(manual=false){const button=$('#check-updates');button.disabled=true;if(manual)$('#update-status').textContent='Checking the official GitHub release…';try{availableUpdate=await invoke('check_for_updates');const install=$('#install-update');install.hidden=!availableUpdate.update_available;if(availableUpdate.update_available){install.textContent=`Install ${availableUpdate.latest_version}`;$('#update-status').textContent=`${availableUpdate.latest_version} is ready${availableUpdate.size_bytes?` · ${Math.round(availableUpdate.size_bytes/1048576)} MB`:''}.`}else{$('#update-status').textContent=`You are up to date · version ${availableUpdate.current_version}.`}}catch(error){availableUpdate=null;$('#install-update').hidden=true;$('#update-status').textContent=`Automatic check failed · ${String(error)}`;if(manual)toast('Update check failed')}finally{button.disabled=false}}
$('#check-updates').onclick=()=>checkForUpdates(true);
$('#install-update').onclick=async event=>{if(!availableUpdate?.package_name)return;const button=event.currentTarget;button.disabled=true;$('#check-updates').disabled=true;$('#update-progress').hidden=false;$('#update-status').textContent='Keep sayIt open while the update is installed.';try{await invoke('install_update',{packageName:availableUpdate.package_name})}catch(error){button.disabled=false;$('#check-updates').disabled=false;$('#update-progress').hidden=true;$('#update-status').textContent=String(error);toast('Update could not be installed')}};
$('#shortcut-form').onsubmit=async event=>{event.preventDefault();const data=new FormData(event.target);try{const settings=await invoke('update_desktop_settings',{settings:{selection_shortcut:String(data.get('selection_shortcut')).trim().toLowerCase(),clipboard_shortcut:String(data.get('clipboard_shortcut')).trim().toLowerCase(),launch_at_login:data.get('launch_at_login')==='on'}});showShortcuts(settings);toast('Shortcut and startup settings updated')}catch(error){toast(String(error))}};
$('#model-import-form').onsubmit=async event=>{event.preventDefault();const data=new FormData(event.target);try{await post('models/imports/local',{directory:data.get('directory'),display_name:data.get('display_name'),license:data.get('license'),license_url:data.get('license_url'),untested_model_and_license_review_confirmed:data.get('acknowledged')==='on'});event.target.reset();toast('Community model imported');refresh('models')}catch(error){toast(String(error))}};
$('#hf-import-form').onsubmit=async event=>{event.preventDefault();const data=new FormData(event.target);try{toast('Resolving immutable repository revision…');await post('models/imports/huggingface',{repository:data.get('repository'),revision:data.get('revision')||null,display_name:data.get('display_name'),license:data.get('license'),license_url:data.get('license_url'),access_token:data.get('access_token')||null,untested_model_and_license_review_confirmed:data.get('acknowledged')==='on'});event.target.reset();toast('Hugging Face model imported');refresh('models')}catch(error){toast(String(error))}};
invoke('desktop_settings').then(showShortcuts).catch(console.error);
setInterval(()=>{const section=document.querySelector('.nav-item.active').dataset.view;refresh(section,section==='models')},15000);
setInterval(pollModelProgress,1000);

playbackView=new window.SayItPlaybackView({post,notify:toast,changed:()=>refresh()});
window.__TAURI__.event.listen('window-visibility',event=>{if(event.payload.label==='main'){visibilityChanged=true;windowVisible=event.payload.visible;playbackView.setVisible(windowVisible);if(windowVisible)refresh(document.querySelector('.nav-item.active').dataset.view)}});
window.__TAURI__.event.listen('show-section',event=>{const button=document.querySelector(`[data-view="${event.payload}"]`);if(button)button.click()});
invoke('desktop_context').then(context=>{if(!visibilityChanged)windowVisible=context.visible;playbackView.setVisible(windowVisible);if(windowVisible)refresh();if(!context.testing){checkForUpdates();showOnboardingWhenNeeded()}});
window.__TAURI__.event.listen('service-event',event=>{const resource=event.payload?.resource;if(!windowVisible||(resource==='playback'&&document.querySelector('.nav-item.active').dataset.view!=='player'))return;if(resource)pendingServiceResources.add(resource);if(serviceEventTimer)return;serviceEventTimer=setTimeout(()=>{serviceEventTimer=null;const section=document.querySelector('.nav-item.active').dataset.view;const details=pendingServiceResources.has(section)||(section==='models'&&pendingServiceResources.has('models'))||(section==='voices'&&(pendingServiceResources.has('voices')||pendingServiceResources.has('models')||pendingServiceResources.has('settings')))||(section==='history'&&(pendingServiceResources.has('history')||pendingServiceResources.has('jobs')))||(section==='settings'&&(pendingServiceResources.has('settings')||pendingServiceResources.has('models')));pendingServiceResources.clear();refresh(section,details)},150)});
window.__TAURI__.event.listen('capture-status',event=>{const status=event.payload;if(status.ok){toast(`${status.kind==='selection'?'Selection':'Clipboard'} sent to speech`);refresh()}else{toast(status.error||'Text could not be captured')}});
window.__TAURI__.event.listen('desktop-action-status',event=>{toast(event.payload.message||'Desktop action completed');refresh()});
window.__TAURI__.event.listen('update-progress',event=>{const progress=event.payload;const percent=progress.total_bytes?Math.min(100,Math.round(progress.downloaded_bytes/progress.total_bytes*100)):0;$('#update-progress').hidden=false;$('#update-progress').classList.toggle('indeterminate',!progress.total_bytes);$('#update-progress').querySelector('span').style.width=`${percent}%`;$('#update-progress-label').textContent=progress.state==='installing'?'Installing and restarting…':progress.total_bytes?`Downloading · ${percent}%`:`Downloading · ${Math.round(progress.downloaded_bytes/1048576)} MB`});
window.__TAURI__.event.listen('long-text-request',async event=>{const request=event.payload;const confirmed=window.confirm(`This ${request.kind} contains ${request.characters.toLocaleString()} characters and may take a while to process. Read it aloud?`);if(!confirmed){toast('Long text was not queued');return}try{await post('jobs',{text:request.text,source:request.kind,queue_policy:'replace',confirmed_long_text:true});toast('Long text queued');refresh()}catch(error){toast(String(error))}});
const onboarding=$('#onboarding');async function showOnboardingWhenNeeded(){try{const hasModel=(await get('models')).some(model=>model.installed);if(!hasModel||!localStorage.getItem('onboarding-complete')){await invoke('open_management',{section:'models'});windowVisible=true;onboarding.showModal()}}catch(error){console.error(error);setTimeout(showOnboardingWhenNeeded,1000)}}onboarding.addEventListener('close',()=>{localStorage.setItem('onboarding-complete','true');document.querySelector('[data-view="models"]').click()});
