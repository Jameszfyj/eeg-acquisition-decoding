'use strict';

// A prompt is acknowledged only after its local audio has ended. Stopped or
// replaced sessions cannot send a delayed acknowledgement or replay old audio.
class CalibrationPrompts {
  constructor(audio, acknowledge, isCurrent, onFailure) {
    this.audio=audio;this.acknowledge=acknowledge;this.isCurrent=isCurrent;this.onFailure=onFailure;
    this.enabled=false;this.prepared=false;this.version=0;this.seen=new Set();this.blocked=new Set();
  }
  async prepare(enabled, tasks) {
    this.cancel();this.enabled=enabled;this.prepared=false;
    const version=this.version;
    if(enabled){
      await this.audio.unlock();
      await this.audio.prepare(['校准完成',...tasks]);
    }
    if(version!==this.version)throw new DOMException('准备已取消','AbortError');
    this.prepared=true;
  }
  async present(c) {
    const key=c.session_id+':'+(c.ready?'complete':c.prompt_id);
    if(this.seen.has(key)||this.blocked.has(c.session_id)||!this.isCurrent(c))return;
    if(!c.ready&&!c.awaiting_display)return;
    this.seen.add(key);
    const version=this.version;
    try{
      if(c.audio_enabled!==this.enabled||this.enabled&&!this.prepared)
        throw new Error('校准语音尚未准备，请停止后重新开始校准。');
      let metadata={enabled:false};
      if(this.enabled){
        const phrase=c.ready?'校准完成':c.phase==='rest'?'休息':c.task;
        const playback=await this.audio.play(phrase);
        await playback.finished;
        metadata=playback.metadata;
      }
      if(version!==this.version||this.blocked.has(c.session_id)||!this.isCurrent(c))return;
      if(!c.ready)await this.acknowledge({prompt_id:c.prompt_id,session_id:c.session_id,audio:metadata});
    }catch(e){
      if(e.name==='AbortError'||version!==this.version||!this.isCurrent(c))return;
      this.cancel(c.session_id);this.onFailure(e);
    }
  }
  cancel(sessionId) {
    this.version++;
    if(sessionId)this.blocked.add(sessionId);
    this.audio.cancel();
  }
}
if(typeof module!=='undefined')module.exports=CalibrationPrompts;
