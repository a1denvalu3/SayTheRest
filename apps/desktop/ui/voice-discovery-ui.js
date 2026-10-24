(function(root){
  const presets={faithful:{temperature:.45,top_p:.75,top_k:20,repetition_penalty:1.3},natural:{temperature:.65,top_p:.9,top_k:40,repetition_penalty:1.2},expressive:{temperature:.85,top_p:.95,top_k:50,repetition_penalty:1.2}};
  const samples={en:'A quiet morning brings a clear new sentence. Every voice gives familiar words its own color, character, and rhythm.',es:'La luz de la mañana entra por la ventana. Cada voz da a estas palabras su propio ritmo y una expresión diferente.',fr:'La lumière du matin éclaire la pièce. Chaque voix donne à ces mots un rythme personnel et une couleur différente.',de:'Das Morgenlicht fällt durch das Fenster. Jede Stimme gibt diesen vertrauten Worten einen eigenen Klang und Rhythmus.',it:'La luce del mattino entra dalla finestra. Ogni voce dà a queste parole un ritmo personale e un colore diverso.',pt:'A luz da manhã entra pela janela. Cada voz dá a estas palavras o seu próprio ritmo e uma expressão diferente.',ru:'Утренний свет наполняет комнату. Каждый голос придаёт этим словам свой ритм и особое настроение.',zh:'清晨的阳光照进房间。每一种声音都能让这些熟悉的话语拥有不同的节奏，色彩和独特的感情。',ja:'朝の光が窓から部屋に入ります。それぞれの声が、聞き慣れた言葉に新しい響きとリズムを与えます。',ko:'아침 햇살이 창문으로 들어옵니다. 각각의 목소리는 익숙한 말에 새로운 느낌과 자신만의 리듬을 더합니다.'};
  const keys=['temperature','top_p','top_k','repetition_penalty'];
  const escape=value=>String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  function languageLabel(code){if(code==='auto')return 'Automatic';try{return new Intl.DisplayNames(navigator.languages,{type:'language'}).of(code)||code}catch{return code}}
  function characterTuning(position){
    const from=position<=.5?presets.faithful:presets.natural,to=position<=.5?presets.natural:presets.expressive;
    const fraction=position<=.5?position*2:(position-.5)*2;
    return Object.fromEntries(keys.map(key=>{const step=key==='top_k'?1:.05;return [key,Number((Math.round((from[key]+(to[key]-from[key])*fraction)/step)*step).toFixed(2))]}));
  }
  function write(form,tuning,position){for(const key of keys)form.elements[key].value=String(tuning[key]);const preset=Object.keys(presets).find(name=>keys.every(key=>presets[name][key]===tuning[key]));form.elements.preset.value=preset||'custom';if(form.elements.character){if(position!==undefined||preset)form.elements.character.value=String(position??({faithful:0,natural:.5,expressive:1}[preset]));form.elements.character.setAttribute('aria-valuetext',preset||'Custom settings')}}
  function read(form){return Object.fromEntries(keys.map(key=>[key,Number(form.elements[key].value)]))}
  function bind(form){form.elements.preset.onchange=()=>{const preset=presets[form.elements.preset.value];if(preset)write(form,preset)};for(const key of keys)form.elements[key].oninput=()=>{form.elements.preset.value='custom';form.elements.character?.setAttribute('aria-valuetext','Custom settings')};if(form.elements.character)form.elements.character.oninput=()=>{const position=Number(form.elements.character.value);write(form,characterTuning(position),position)}}
  class VoiceDiscoveryView{
    constructor({get,post,del,notify,changed,visible}){
      Object.assign(this,{get,post,del,notify,changed,visible});
      this.panel=document.querySelector('#voice-discovery');this.form=document.querySelector('#discovery-form');this.cards=document.querySelector('#discovery-candidates');
      this.editor=document.querySelector('#voice-generation-form');this.dialog=document.querySelector('#voice-generation-dialog');
      this.profiles=[];this.settings={};this.snapshot=null;this.inFlight=false;this.pending=false;this.generating=false;
      write(this.form,presets.natural);bind(this.form);bind(this.editor);
      this.form.onsubmit=event=>{event.preventDefault();this.generate()};
      this.form.elements.language.onchange=()=>this.defaultText();
      this.form.elements.randomize_tuning.onchange=()=>this.syncControls();
      document.querySelector('#discovery-cancel').onclick=()=>this.action(()=>this.post('voices/discovery/cancel',{}));
      document.querySelector('#discovery-clear').onclick=()=>this.action(()=>this.del('voices/discovery'));
      document.querySelector('#cancel-voice-generation').onclick=()=>this.dialog.close();
      this.editor.onsubmit=async event=>{event.preventDefault();const button=this.editor.querySelector('[type=submit]');button.disabled=true;try{await this.post(`voices/${this.editor.dataset.voice}/generation`,read(this.editor));this.dialog.close();this.notify('Voice settings saved');await this.changed()}catch(error){this.notify(String(error))}finally{button.disabled=false}};
      setInterval(()=>{if(this.generating&&this.visible()&&!this.panel.hidden)this.refresh()},1000);
    }
    defaultText(){const field=this.form.elements.sample_text;if(!field.value||field.value===this.lastDefault){this.lastDefault=samples[this.form.elements.language.value]||samples.en;field.value=this.lastDefault}}
    render(model,profiles,settings){
      this.profiles=profiles;this.settings=settings;this.panel.hidden=!model?.capabilities.voice_discovery;
      if(this.panel.hidden)return;
      if(this.model!==model.id){this.model=model.id;this.cards.replaceChildren();this.form.elements.language.innerHTML=model.languages.map(code=>`<option value="${escape(code)}">${escape(languageLabel(code))}</option>`).join('');this.form.elements.language.value=model.selected_language||'auto';this.form.elements.sample_text.value='';this.defaultText()}
      if(this.snapshot)this.renderSnapshot(this.snapshot);
      this.refresh();
    }
    async refresh(){
      if(this.panel.hidden||!this.visible())return;
      if(this.inFlight){this.pending=true;return}
      this.inFlight=true;
      try{this.snapshot=await this.get('voices/discovery');this.renderSnapshot(this.snapshot)}catch(error){this.notify(String(error))}
      finally{this.inFlight=false;if(this.pending){this.pending=false;this.refresh()}}
    }
    renderSnapshot(snapshot){
      if(snapshot?.model_id!==this.model)snapshot=null;
      this.generating=snapshot?.state==='generating';
      this.syncControls();
      this.form.querySelector('[type=submit]').disabled=this.generating;
      document.querySelector('#discovery-cancel').hidden=!this.generating;
      document.querySelector('#discovery-clear').hidden=this.generating||!snapshot;
      const state=snapshot?.state;
      document.querySelector('#discovery-status').textContent=this.generating?`${snapshot.completed} of ${snapshot.total} new samples ready…`:state==='failed'?snapshot.error:state==='cancelled'?'Generation cancelled. Completed samples are available.':snapshot?.candidates.length?`${snapshot.candidates.length} ${snapshot.candidates.length===1?'sample':'samples'} to explore. Save favorites before restarting.`:'';
      const candidates=snapshot?.candidates||[],ids=new Set(candidates.map(candidate=>candidate.id));
      for(const card of [...this.cards.children])if(!ids.has(card.dataset.candidate))card.remove();
      for(const candidate of candidates){
        let card=this.cards.querySelector(`[data-candidate="${candidate.id}"]`);
        if(!card){
          card=document.createElement('article');card.className='data-card';card.dataset.candidate=candidate.id;
          card.innerHTML=`<header><h3>Voice sample ${candidates.indexOf(candidate)+1}</h3><span>${escape(candidate.language)} · ${candidate.duration_seconds.toFixed(1)} sec</span></header><p>${escape(candidate.sample_text)}</p><small>Seed ${candidate.seed}</small><label>Voice name<input data-candidate-name maxlength="80" required value="New voice ${escape(candidate.id.slice(0,4))}"></label><p data-save-error></p><div class="card-actions"><button type="button" class="action secondary" data-candidate-action="preview">Preview</button><button type="button" class="action" data-candidate-action="save">Save voice</button><button type="button" class="action secondary" data-candidate-action="select" hidden>Use this voice</button></div>`;
          this.cards.append(card);
          for(const button of card.querySelectorAll('[data-candidate-action]'))button.onclick=()=>this.candidateAction(card,button);
        }
        const saved=this.profiles.find(profile=>profile.id===candidate.saved_voice_profile_id);
        card.dataset.saved=saved?.id||'';
        const save=card.querySelector('[data-candidate-action=save]');save.disabled=Boolean(saved||candidate.save_error);save.textContent=saved?'Saved':'Save voice';
        card.querySelector('[data-save-error]').textContent=candidate.save_error||'';
        const select=card.querySelector('[data-candidate-action=select]');select.hidden=!saved;select.disabled=saved?.id===this.settings.active_voice_profile_id;select.textContent=select.disabled?'In use':'Use this voice';
      }
    }
    syncControls(){for(const field of this.form.querySelectorAll('[name]'))field.disabled=this.generating;const random=this.form.elements.randomize_tuning.checked;for(const name of [...keys,'character','preset'])this.form.elements[name].disabled=this.generating||random}
    async generate(){
      const button=this.form.querySelector('[type=submit]');button.disabled=true;
      try{const seed=this.form.elements.seed.value,random=this.form.elements.randomize_tuning.checked;await this.post('voices/discovery',{sample_text:this.form.elements.sample_text.value,language:this.form.elements.language.value,candidate_count:Number(this.form.elements.candidate_count.value),seed:seed===''?null:Number(seed),randomize_tuning:random?true:undefined,tuning:random?undefined:read(this.form)});await this.refresh()}catch(error){this.notify(String(error));button.disabled=false}
    }
    async action(run){try{await run();await this.refresh()}catch(error){this.notify(String(error))}}
    async candidateAction(card,button){
      button.disabled=true;
      try{
        const action=button.dataset.candidateAction,id=card.dataset.candidate;
        if(action==='preview')await this.post(`voices/discovery/${id}/preview`,{});
        else if(action==='select'){await this.post(`voices/${card.dataset.saved}/select`,{});await this.changed()}
        else{const name=card.querySelector('[data-candidate-name]');if(!name.reportValidity())return;await this.post(`voices/discovery/${id}/save`,{name:name.value});this.notify('Voice saved');await this.changed()}
      }catch(error){this.notify(String(error))}
      finally{button.disabled=false;await this.refresh()}
    }
    openTuning(voice){this.editor.dataset.voice=voice.id;write(this.editor,voice.generation_tuning||{temperature:.9,top_p:1,top_k:50,repetition_penalty:1.05});this.dialog.showModal()}
  }
  root.VoiceDiscoveryView=VoiceDiscoveryView;
})(window);
