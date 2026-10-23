(function (root) {
  'use strict';
  function tokenize(text) {
    const offsets = new Map();
    let byte = 0, character = 0;
    for (const value of text) { offsets.set(byte, character++); byte += value.length; }
    offsets.set(byte, character);
    const segments = typeof Intl.Segmenter === 'function'
      ? Array.from(new Intl.Segmenter(undefined, {granularity:'word'}).segment(text)).filter(part => part.isWordLike)
      : Array.from(text.matchAll(/\S+/gu), match => ({segment:match[0], index:match.index}));
    return segments.map(part => ({text:part.segment, index:part.index,
      start:offsets.get(part.index), end:offsets.get(part.index + part.segment.length)}));
  }

  function durationForChunk(chunks, index) {
    const chunk = chunks[index];
    if (Number.isFinite(chunk.audio_end_seconds)) return Math.max(0, chunk.audio_end_seconds - chunk.audio_start_seconds);
    if (index + 1 < chunks.length) return Math.max(0, chunks[index + 1].audio_start_seconds - chunk.audio_start_seconds);
    let charactersPerSecond = 15;
    const previous = chunks[index - 1];
    if (previous && previous.audio_end_seconds > previous.audio_start_seconds) {
      charactersPerSecond = Math.min(30, Math.max(5,
        (previous.text_end - previous.text_start) / (previous.audio_end_seconds - previous.audio_start_seconds)));
    }
    return Math.max(1, chunk.text_end - chunk.text_start) / charactersPerSecond;
  }

  function timings(words, chunks, duration, complete) {
    if (!chunks.length) {
      const end = words.at(-1)?.end || 0;
      return words.map(word => complete && duration > 0 && end > 0 ? word.start / end * duration : Infinity);
    }
    let index = 0;
    return words.map(word => {
      while (index + 1 < chunks.length && chunks[index + 1].text_start <= word.start) index++;
      const chunk = chunks[index];
      if (word.start < chunk.text_start || word.start >= chunk.text_end) return Infinity;
      const span = durationForChunk(chunks, index);
      return chunk.audio_start_seconds + (word.start - chunk.text_start) / Math.max(1, chunk.text_end - chunk.text_start) * span;
    });
  }

  function activeWord(times, seconds, generatedDuration, chunks = []) {
    if (!Number.isFinite(seconds) || seconds < 0 || seconds >= generatedDuration) return -1;
    if (chunks.length) {
      const chunk = chunks.findLast(part => part.audio_start_seconds <= seconds);
      if (!chunk || (Number.isFinite(chunk.audio_end_seconds) && seconds >= chunk.audio_end_seconds)) return -1;
    }
    let low = 0, high = times.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (times[middle] <= seconds) low = middle + 1; else high = middle;
    }
    return low - 1;
  }

  function estimatedDuration(text, chunks, generatedDuration, complete) {
    if (complete) return generatedDuration;
    const characters = Array.from(text).length;
    const last = chunks.at(-1);
    const rate = last?.audio_end_seconds > 0 && last.text_end > 0
      ? Math.min(30, Math.max(5, last.text_end / last.audio_end_seconds)) : 15;
    return Math.max(generatedDuration, characters / rate);
  }

  const api = {tokenize, timings, activeWord, estimatedDuration};
  root.SayItReaderTiming = api;
  if (typeof module !== 'undefined') module.exports = api;
})(typeof window === 'undefined' ? globalThis : window);
