"use strict";
// Audio is decoded before the device session starts; no speech is fetched mid-trial.
class CollectionAudio {
  constructor(api, status) {
    this.api = api;
    this.status = status;
    this.context = null;
    this.buffers = new Map();
    this.generation = 0;
    this.pending = Promise.resolve();
    this.active = null;
    this.volume = 0.7;
  }
  async unlock() {
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) throw new Error("此浏览器不支持声音提示，请换用Chrome或Edge，或关闭声音。");
    if (!this.context || this.context.state === "closed") this.context = new Context();
    await this.context.resume();
    if (this.context.state !== "running") throw new Error("浏览器尚未允许播放声音，请先点击试听。");
  }
  async prepare(texts) {
    const version = this.generation;
    const wanted = [...new Set(["开始", "休息", "结束", ...texts.map(t => t.trim())])];
    this.status("正在准备本机中文语音…");
    const manifest = await this.api("/api/audio/prepare", {texts: wanted});
    const buffers = new Map();
    await Promise.all(manifest.cues.map(async cue => {
      const response = await fetch(cue.url);
      if (!response.ok) throw new Error("无法读取提示音，请重新试听。");
      buffers.set(cue.text, await this.context.decodeAudioData(await response.arrayBuffer()));
    }));
    if (version !== this.generation) throw new DOMException("播放已取消", "AbortError");
    for (const text of wanted) if (!buffers.has(text)) throw new Error("提示音缺少任务：" + text);
    this.buffers = buffers;
    this.status("本机中文语音已就绪");
  }
  async play(text, kind = "speech", options = {}) {
    const version = this.generation;
    await this.pending;
    if (version !== this.generation) throw new DOMException("播放已取消", "AbortError");
    if (!this.context || this.context.state !== "running") throw new Error("声音播放已暂停，采集不能继续。");
    let buffer;
    if (kind === "beep") {
      const rate = this.context.sampleRate, length = Math.round(rate * 0.4);
      buffer = this.context.createBuffer(1, length, rate);
      const samples = buffer.getChannelData(0);
      for (let i = 0; i < length; i++) {
        const fade = Math.min(1, i / (rate * 0.005), (length - 1 - i) / (rate * 0.005));
        samples[i] = Math.sin(2 * Math.PI * 880 * i / rate) * 0.24 * fade;
      }
    } else {
      buffer = this.buffers.get(text);
      if (!buffer) throw new Error("缺少任务语音：" + text + "。请重新开始并准备声音。");
    }
    const rate=options.rate??1;
    if(!Number.isFinite(rate)||rate<.5||rate>2)throw new Error('语音播放速度无效。');
    let offset=0,segmentDuration=buffer.duration;
    if(options.trim){
      const channels=Array.from({length:buffer.numberOfChannels},(_,i)=>buffer.getChannelData(i));
      const levels=new Float32Array(buffer.length);let peak=0;
      for(let i=0;i<levels.length;i++){
        for(const samples of channels)levels[i]=Math.max(levels[i],Math.abs(samples[i]));
        peak=Math.max(peak,levels[i]);
      }
      if(!peak)throw new Error('动作语音没有有效声音，请重新准备。');
      const threshold=peak*.01,padding=Math.round(buffer.sampleRate*.08);
      let first=0,last=levels.length-1;
      while(first<last&&levels[first]<=threshold)first++;
      while(last>first&&levels[last]<=threshold)last--;
      first=Math.max(0,first-padding);last=Math.min(levels.length,last+1+padding);
      offset=first/buffer.sampleRate;segmentDuration=(last-first)/buffer.sampleRate;
    }
    const duration=segmentDuration/rate;
    const source = this.context.createBufferSource(), gain = this.context.createGain();
    source.buffer = buffer; gain.gain.value = this.volume;
    if(rate!==1)source.playbackRate.value=rate;
    source.connect(gain); gain.connect(this.context.destination);
    const metadata = {enabled: true, engine: "local_wav_webaudio", kind, text,
      scheduled_browser_ms: performance.timeOrigin + performance.now(),
      duration_seconds: duration};
    if(options.trim||rate!==1)Object.assign(metadata,{playback_rate:rate,audio_offset_seconds:offset});
    let rejectPlayback;
    const finished = new Promise((resolve, reject) => {
      rejectPlayback = reject;
      const timeout = setTimeout(() => {
        source.onended = null;
        try { source.stop(); } catch (_) {}
        this.active = null;
        reject(new Error("提示音播放没有正常结束，采集已停止。"));
      }, duration * 1000 + 3000);
      source.onended = () => {
        clearTimeout(timeout); this.active = null;
        metadata.completed_browser_ms = performance.timeOrigin + performance.now();
        source.disconnect(); gain.disconnect(); resolve(metadata);
      };
      this.active = {source, timeout, reject};
    });
    // Attach a rejection handler immediately; the owner still observes failure.
    this.pending = finished;
    finished.catch(() => {});
    try { if(options.trim)source.start(0,offset,segmentDuration);else source.start(); }
    catch (error) {
      const active = this.active; this.active = null;
      clearTimeout(active.timeout); source.onended = null; rejectPlayback(error);
      throw error;
    }
    this.status(kind === "beep" ? "提示音：滴（0.4秒）· 开始想象" : "语音：" + text);
    return {metadata, finished};
  }
  cancel() {
    this.generation++;
    if (this.active) {
      const active = this.active; this.active = null;
      clearTimeout(active.timeout); active.source.onended = null;
      try { active.source.stop(); active.source.disconnect(); } catch (_) {}
      active.reject(new DOMException("播放已取消", "AbortError"));
    }
    this.pending = Promise.resolve();
  }
}
class StaticSubjectView {
  constructor(view, textNode, doc = document, roundsNode = null) {
    this.view = view; this.textNode = textNode; this.doc = doc; this.visible = false;
    this.roundsNode = roundsNode;
    doc.addEventListener("fullscreenchange", () => {
      if (this.visible && doc.fullscreenElement !== this.view) this.hide();
    });
    // Embedded browsers may send Escape to the page instead of leaving native fullscreen.
    doc.addEventListener("keydown", event => {
      if (event.key === "Escape" && this.visible) {
        event.preventDefault();
        return this.leave().catch(() => this.hide());
      }
    });
  }
  setText(text) {
    // Do not replace the text node on prepare -> task or during state polling.
    if (this.textNode.textContent !== text) this.textNode.textContent = text;
  }
  setRounds(text) {
    if (!this.roundsNode) return;
    if (this.roundsNode.textContent !== text) this.roundsNode.textContent = text;
    this.roundsNode.hidden = !text;
  }
  update(state) {
    const phase = state.phase;
    let text = "休息";
    if (state.running && phase && ["prepare", "task"].includes(phase.phase))
      text = phase.prompt_text || phase.task_text;
    else if ((state.running && phase?.phase === "finishing") || state.status === "completed")
      text = "结束";
    else if (["failed", "cancelled", "stopping"].includes(state.status))
      text = "停止";
    this.setText(text || "休息");
    // Round feedback is static and only appears after the last task in a round.
    let rounds = "";
    if (state.status === "completed") rounds = "还剩 0 轮";
    else if (state.running && phase?.round_complete &&
             ["rest", "block_rest", "finishing"].includes(phase.phase) &&
             Number.isInteger(phase.remaining_rounds) && phase.remaining_rounds >= 0)
      rounds = `还剩 ${phase.remaining_rounds} 轮`;
    this.setRounds(rounds);
  }
  async enter() {
    this.visible = true; this.view.hidden = false;
    this.doc.body.classList.add("subject-mode");
    try {
      if (this.doc.fullscreenElement !== this.view) {
        if (!this.view.requestFullscreen) throw new Error("浏览器不支持全屏");
        await this.view.requestFullscreen({navigationUI:"hide"});
      }
      if (this.doc.fullscreenElement !== this.view) throw new Error("浏览器未进入全屏");
    } catch (error) {
      this.hide();
      throw new Error("未能进入被试全屏，请在Chrome或Edge中点击开始后重试。");
    }
  }
  hide() {
    this.visible = false; this.view.hidden = true;
    this.doc.body.classList.remove("subject-mode");
  }
  async leave() {
    try {
      if (this.doc.fullscreenElement === this.view) await this.doc.exitFullscreen();
    } finally { this.hide(); }
  }
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = CollectionAudio;
  module.exports.StaticSubjectView = StaticSubjectView;
} else {
  window.CollectionAudio = CollectionAudio;
  window.StaticSubjectView = StaticSubjectView;
}
