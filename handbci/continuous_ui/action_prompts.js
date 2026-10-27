'use strict';

// Speak the requested hand action once at its first visible cue, never a decoded result.
class ActionPrompts {
  static names={thumb:'拇指',index:'食指',middle:'中指',ring:'无名指',little:'小指',pinch:'OK手势'};
  constructor(audio,isCurrent,onFailure){
    this.audio=audio;this.isCurrent=isCurrent;this.onFailure=onFailure;
    this.prepared=false;this.version=0;this.seen=new Set();this.blocked=new Set();
  }
  async prepare(){
    this.cancel();this.prepared=false;const version=this.version;
    await this.audio.unlock();await this.audio.prepare(Object.values(ActionPrompts.names));
    if(version!==this.version)throw new DOMException('语音准备已取消','AbortError');
    this.prepared=true;
  }
  async present(p){
    if(p.phase!=='awaiting_display'||!p.prompt_id||this.blocked.has(p.session_id)||!this.isCurrent(p))return null;
    const key=p.session_id+':'+p.prompt_id;
    if(this.seen.has(key))return null;
    if(!this.prepared)throw new Error('动作语音未就绪，请停止后重新开始解码。');
    const phrase=ActionPrompts.names[p.target_action];
    if(!phrase)throw new Error('当前目标没有对应的机械手动作语音。');
    this.seen.add(key);const version=this.version;
    // Use the original full WAV at its natural speed, matching calibration speech.
    const playback=await this.audio.play(phrase);
    playback.finished.catch(error=>{
      if(error.name!=='AbortError'&&version===this.version&&this.isCurrent(p))this.onFailure(error);
    });
    // Display acknowledgement does not wait for the speech to finish.
    if(version!==this.version||!this.isCurrent(p))return null;
    return playback.metadata;
  }
  cancel(sessionId){
    this.version++;if(sessionId)this.blocked.add(sessionId);this.audio.cancel();
  }
}
if(typeof module!=='undefined')module.exports=ActionPrompts;
