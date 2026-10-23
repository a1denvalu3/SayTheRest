(function (root) {
  'use strict';
  const timing = root.SayItReaderTiming;
  const clock = seconds => {
    const value = Math.max(0, Math.floor(seconds || 0));
    return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`;
  };
  class PlaybackView {
    constructor({post, notify = () => {}, changed = () => {}}) {
      this.$ = selector => document.querySelector(selector);
      this.post = post; this.notify = notify; this.changed = changed;
      this.words = []; this.nodes = []; this.times = []; this.current = -1;
      this.following = true; this.visible = true; this.scrubbing = false;
      this.viewport = this.$('#reader-viewport');
      this.reader = this.$('#spoken-text');
      this.$('#follow-reader').onclick = () => { this.following = true; this.followWord(); this.followButton(); };
      this.viewport.addEventListener('wheel', () => this.unfollow(), {passive:true});
      this.viewport.addEventListener('touchstart', () => this.unfollow(), {passive:true});
      this.viewport.addEventListener('pointerdown', event => {
        if (event.target === this.viewport) this.unfollow();
      });
      this.viewport.addEventListener('keydown', event => {
        if (['PageUp','PageDown','Home','End','ArrowUp','ArrowDown'].includes(event.key)) this.unfollow();
        const index = Number(event.target.dataset.word);
        if (Number.isInteger(index) && ['ArrowLeft','ArrowRight'].includes(event.key)) {
          event.preventDefault();
          const next = Math.min(this.nodes.length - 1, Math.max(0, index + (event.key === 'ArrowRight' ? 1 : -1)));
          this.nodes[next]?.focus();
        }
        if (event.key === 'Enter' || event.key === ' ') {
          if (Number.isInteger(index)) { event.preventDefault(); this.seekWord(index); }
        }
      });
      this.reader.addEventListener('click', event => {
        const node = event.target.closest('[data-word]');
        if (node && !root.getSelection()?.toString()) this.seekWord(Number(node.dataset.word));
      });
      this.$('#play-toggle').onclick = () => this.command(this.playback?.state === 'playing' ? 'pause' : 'resume');
      this.$('[data-control=seek-back]').onclick = () => this.seek(this.position() - 15);
      this.$('[data-control=seek-forward]').onclick = () => this.seek(this.position() + 30);
      this.$('[data-control=stop]').onclick = () => this.command('stop');
      const range = this.$('#timeline');
      range.oninput = () => { this.scrubbing = true; this.updateClock(Number(range.value)); };
      range.onchange = async () => { await this.seek(Number(range.value)); this.scrubbing = false; };
      range.onblur = () => { this.scrubbing = false; };
      this.$('#rate').onchange = event => this.command('rate', {rate:Number(event.target.value)});
      this.$('#volume').onchange = event => this.command('volume', {volume:Number(event.target.value)});
      this.$('#rate').oninput = event => { this.$('#rate-value').textContent=`${Number(event.target.value)}×`; };
      this.$('#volume').oninput = event => { this.$('#volume-value').textContent=`${Math.round(Number(event.target.value)*100)}%`; };
      for (const name of ['rate','volume']) {
        this.$(`#${name}-toggle`).onclick = () => {
          const pane = this.$(`#${name}-pane`);
          pane.hidden = !pane.hidden;
          this.$(`#${name}-toggle`).setAttribute('aria-expanded', String(!pane.hidden));
          const other = name === 'rate' ? 'volume' : 'rate';
          this.$(`#${other}-pane`).hidden = true;
          this.$(`#${other}-toggle`).setAttribute('aria-expanded', 'false');
          if (!pane.hidden) this.$(`#${name}`).focus();
        };
      }
      this.timer = setInterval(() => { if (this.visible && this.playback && !this.scrubbing) this.updateClock(this.position()); }, 100);
      new ResizeObserver(() => this.drawWaveform(this.position())).observe(this.$('#waveform'));
    }
    async command(action, body = {}) {
      try { await this.post(`playback/${action}`, body); this.changed(); }
      catch (error) { this.notify(String(error)); }
    }
    async seek(seconds) {
      if (!this.playback?.job_id) return;
      await this.command('seek', {seconds:Math.max(0, Math.min(this.playback.duration_seconds, seconds))});
    }
    seekWord(index) {
      const seconds = this.times[index];
      if (!Number.isFinite(seconds) || seconds > this.playback.duration_seconds) return;
      this.following = true; this.followButton(); this.seek(seconds);
    }
    unfollow() { this.following = false; this.followButton(); }
    followButton() { this.$('#follow-reader').hidden = this.following || this.current < 0; }
    followWord() {
      if (!this.following || this.current < 0 || !this.visible) return;
      const word = this.nodes[this.current], viewport = this.viewport.getBoundingClientRect();
      if (!word || viewport.height === 0) return;
      const bounds = word.getBoundingClientRect();
      if (bounds.top < viewport.top + 18 || bounds.bottom > viewport.bottom - 18) {
        this.viewport.scrollTop += bounds.top - viewport.top - viewport.height * 0.4;
      }
    }
    position() {
      const p = this.playback;
      if (!p) return 0;
      const elapsed = p.state === 'playing' && !p.buffering ? (performance.now() - this.received) / 1000 * p.rate : 0;
      return Math.min(p.duration_seconds || 0, Math.max(0, p.position_seconds + elapsed));
    }
    render(playback, queueDepth = 0) {
      this.playback = playback; this.received = performance.now();
      const p = playback, present = Boolean(p.job_id), active = present && ['playing','paused','completed'].includes(p.state);
      const generating = present && (p.state === 'synthesizing' || p.generation_complete === false);
      this.$('#now-title').textContent = p.current_title || '';
      this.$('#now-title').hidden = !p.current_title;
      this.$('#play-toggle').disabled = !active;
      this.$('#play-toggle').dataset.playing = String(p.state === 'playing');
      this.$('#play-toggle').setAttribute('aria-label', p.state === 'playing' ? 'Pause' : p.state === 'completed' ? 'Replay' : 'Play');
      this.$('[data-control=stop]').disabled = !present;
      this.$('[data-control=stop]').setAttribute('aria-label', generating || p.state === 'playing' ? 'Stop speech' : 'Clear recording');
      for (const button of document.querySelectorAll('[data-control=seek-back],[data-control=seek-forward]')) button.disabled = !active;
      this.$('#timeline').disabled = !active;
      const status = p.state === 'paused' ? 'Paused' : p.state === 'completed' ? 'Finished' : p.buffering ? (p.duration_seconds > 0 ? 'Buffering…' : 'Preparing speech…') : p.state === 'playing' ? 'Playing' : 'Ready to read';
      const statusText = queueDepth ? `${status} · ${queueDepth} queued` : status;
      if(this.$('#playback-status').textContent!==statusText)this.$('#playback-status').textContent=statusText;
      this.$('#generation-indicator').hidden = !generating;
      const generationText = p.duration_seconds > 0 ? 'Generating the next part locally' : 'Generating speech locally';
      if(this.$('#generation-detail').textContent!==generationText)this.$('#generation-detail').textContent=generationText;
      if (document.activeElement !== this.$('#rate')) this.$('#rate').value = p.rate;
      if (document.activeElement !== this.$('#volume')) this.$('#volume').value = p.volume;
      this.$('#rate-value').textContent = `${Number(p.rate.toFixed(2))}×`;
      this.$('#rate-toggle').textContent = `${Number(p.rate.toFixed(2))}×`;
      this.$('#volume-value').textContent = `${Math.round(p.volume * 100)}%`;
      const documentKey = `${p.job_id || ''}\0${p.spoken_text || ''}`;
      if (documentKey !== this.documentKey) {
        this.documentKey = documentKey; this.current = -1; this.following = true; this.viewport.scrollTop = 0;
        this.words = timing.tokenize(p.spoken_text || ''); this.nodes = []; this.reader.replaceChildren();
        let cursor = 0;
        const fragment = document.createDocumentFragment();
        this.words.forEach((word, index) => {
          fragment.append(document.createTextNode(p.spoken_text.slice(cursor, word.index)));
          const span = document.createElement('span'); span.className = 'reader-word';
          span.dataset.word = index; span.tabIndex = index === 0 ? 0 : -1; span.setAttribute('role','button');
          span.textContent = word.text; fragment.append(span); this.nodes.push(span);
          cursor = word.index + word.text.length;
        });
        fragment.append(document.createTextNode(p.spoken_text ? p.spoken_text.slice(cursor) : 'Select text in another app or read your clipboard.'));
        this.reader.append(fragment); this.timingKey = null;
      }
      const chunks = p.spoken_chunks || [];
      const timingKey = `${documentKey}|${JSON.stringify(chunks)}|${!chunks.length ? p.duration_seconds : ''}`;
      if (timingKey !== this.timingKey) {
        this.timingKey = timingKey; this.times = timing.timings(this.words, chunks, p.duration_seconds, p.generation_complete !== false);
        this.nodes.forEach((node, index) => {
          const enabled = Number.isFinite(this.times[index]) && this.times[index] <= p.duration_seconds;
          node.setAttribute('aria-disabled', String(!enabled));
          node.title = enabled ? `Seek to ${clock(this.times[index])}` : 'Speech has not been generated yet';
        });
      }
      this.nodes.forEach((node,index)=>{const disabled=String(!Number.isFinite(this.times[index])||this.times[index]>p.duration_seconds);if(node.getAttribute('aria-disabled')!==disabled)node.setAttribute('aria-disabled',disabled)});
      this.estimated = timing.estimatedDuration(p.spoken_text || '', chunks, p.duration_seconds, p.generation_complete !== false);
      this.$('#reader-section').hidden = !present;
      if (!this.scrubbing) this.updateClock(p.position_seconds);
    }
    updateClock(seconds) {
      if (!this.playback) return;
      const p = this.playback, range = this.$('#timeline');
      range.max = p.duration_seconds || 0;
      if (!this.scrubbing) range.value = Math.min(seconds, p.duration_seconds || 0);
      range.style.width = `${this.estimated > 0 ? Math.min(100, p.duration_seconds / this.estimated * 100) : 100}%`;
      this.$('#position').textContent = clock(seconds);
      this.$('#duration').textContent = `${p.generation_complete === false ? '~' : ''}${clock(this.estimated)}`;
      const index = p.state === 'completed' ? this.words.length - 1 : timing.activeWord(this.times, seconds, p.duration_seconds, p.spoken_chunks || []);
      if (index !== this.current) {
        for (let i = Math.max(0, Math.min(this.current, index)); i <= Math.max(this.current, index); i++) {
          const node = this.nodes[i]; if (!node) continue;
          node.classList.toggle('current', i === index); node.classList.toggle('spoken', i < index);
          if (i === index) node.setAttribute('aria-current','true'); else node.removeAttribute('aria-current');
          node.tabIndex = i === index || (index < 0 && i === 0) ? 0 : -1;
        }
        this.current = index; this.followWord(); this.followButton();
      }
      this.drawWaveform(seconds);
    }
    drawWaveform(seconds) {
      const canvas = this.$('#waveform'), p = this.playback;
      if (!p || !this.visible || canvas.clientWidth === 0) return;
      const ratio = root.devicePixelRatio || 1, width = canvas.clientWidth, height = canvas.clientHeight;
      if (canvas.width !== width * ratio || canvas.height !== height * ratio) { canvas.width = width * ratio; canvas.height = height * ratio; }
      const ctx = canvas.getContext('2d'); ctx.setTransform(ratio,0,0,ratio,0,0); ctx.clearRect(0,0,width,height);
      const style = getComputedStyle(canvas), accent = style.getPropertyValue('--playback-accent').trim() || '#087aff';
      const muted = style.getPropertyValue('--wave-muted').trim() || '#c6c8cb';
      const amplitudes = p.amplitudes || [], peak = Math.max(0.0001, ...amplitudes);
      const generatedWidth = this.estimated > 0 ? Math.min(1, p.duration_seconds / this.estimated) * width : 0;
      const count = Math.max(1, Math.floor(generatedWidth / 4));
      for (let bar = 0; bar < count && amplitudes.length; bar++) {
        const begin = Math.floor(bar / count * amplitudes.length), end = Math.max(begin + 1, Math.floor((bar + 1) / count * amplitudes.length));
        let level = 0; for (let i = begin; i < end; i++) level = Math.max(level, amplitudes[i] || 0);
        const x = bar * 4, barHeight = Math.max(2, level / peak * (height - 8));
        ctx.fillStyle = x < (p.duration_seconds ? seconds / p.duration_seconds * generatedWidth : 0) ? accent : muted;
        ctx.fillRect(x, (height - barHeight) / 2, 2.5, barHeight);
      }
      if (p.generation_complete === false || !amplitudes.length) {
        ctx.fillStyle = muted;
        for (let x = generatedWidth; x < width; x += 6) ctx.fillRect(x, height / 2 - 1, 2, 2);
      }
    }
    setVisible(visible) { this.visible = visible; if (visible && this.playback) this.updateClock(this.position()); }
  }
  root.SayItPlaybackView = PlaybackView;
})(window);
