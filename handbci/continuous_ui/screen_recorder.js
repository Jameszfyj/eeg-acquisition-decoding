'use strict';

// Capture is requested only from a Start click in the window owning this recording. Video chunks
// are uploaded sequentially to localhost rather than kept for the whole run.
class ScreenRecorder {
  constructor({api,upload,capture,Recorder,onStatus=()=>{}}){
    this.api=api;this.upload=upload;this.capture=capture;this.Recorder=Recorder;this.onStatus=onStatus;
    this.stream=null;this.recorder=null;this.record=null;this.owner=null;this.queue=Promise.resolve();
    this.sequence=0;this.queued=0;this.error='';this.finishing=null;this.version=0;
  }
  status(phase,message,extra={}){this.onStatus({phase,message,...extra});}
  prepare(){
    if(!this.capture||!this.Recorder)return Promise.reject(new Error('当前浏览器不支持录屏，请用Chrome或Edge打开平台，或取消录屏。'));
    const version=++this.version;
    this.finishing=null;this.error='';this.record=null;this.queue=Promise.resolve();this.sequence=0;this.queued=0;
    this.status('selecting','请选择主屏；需要语音时勾选共享系统音频。');
    // Do not await anything before this call: browsers require a fresh click.
    let request;
    try{request=this.capture({video:{displaySurface:'monitor',frameRate:30,width:{ideal:1920},height:{ideal:1080}},audio:true,systemAudio:'include',selfBrowserSurface:'include'});}
    catch(error){request=Promise.reject(error);}
    return Promise.resolve(request).then(stream=>{
      if(version!==this.version){stream.getTracks().forEach(t=>t.stop());throw new Error('录屏准备已取消。');}
      if(!stream.getVideoTracks().length){stream.getTracks().forEach(t=>t.stop());throw new Error('没有获取到屏幕画面。');}
      this.stream=stream;
      stream.getVideoTracks()[0].addEventListener('ended',()=>{
        if(this.stream===stream){this.error='屏幕共享已提前停止';this.stop({complete:false,reason:this.error});}
      });
      this.status('ready',stream.getAudioTracks().length?'屏幕与系统声音已就绪':'屏幕已就绪 · 未选中系统音频，本次视频无声音');
    }).catch(error=>{
      this.status('error','未开始录屏：'+(error.name==='NotAllowedError'?'屏幕选择已取消或未获授权':error.message));
      throw error;
    });
  }
  async start(owner){
    if(!this.stream||!this.stream.getVideoTracks().some(t=>t.readyState!=='ended'))throw new Error('屏幕共享已停止，请重新开始并选择主屏。');
    this.owner=owner;
    const startVersion=this.version,stream=this.stream;
    const hasAudio=this.stream.getAudioTracks().length>0;
    const candidates=hasAudio?['video/webm;codecs=vp8,opus','video/webm','video/mp4']:['video/webm;codecs=vp8','video/webm','video/mp4'];
    const mime=candidates.find(t=>this.Recorder.isTypeSupported(t));
    if(!mime)throw new Error('浏览器没有可用的视频编码器，请取消录屏或更换浏览器。');
    const recorder=new this.Recorder(this.stream,{mimeType:mime,videoBitsPerSecond:5000000});
    const record=await this.api('/api/recording/open',{subject_window_id:owner,mime_type:recorder.mimeType||mime,
      has_audio:hasAudio,capture_surface:this.stream.getVideoTracks()[0].getSettings?.().displaySurface||'unknown'});
    if(startVersion!==this.version||!this.stream){
      stream.getTracks().forEach(track=>track.stop());
      await this.api('/api/recording/finish',{recording_id:record.id,subject_window_id:owner,chunks:0,complete:false,reason:'开始录屏前已取消'});
      throw new Error('录屏准备已取消。');
    }
    this.record=record;this.recorder=recorder;
    this.stopped=new Promise(resolve=>{recorder.onstop=resolve;});
    recorder.ondataavailable=event=>{
      if(!event.data?.size)return;
      const sequence=this.sequence++;this.queued++;
      this.queue=this.queue.then(async()=>{
        if(this.error&&this.uploadFailed)return;
        await this.upload(this.record.id,this.owner,sequence,event.data);
      }).catch(error=>{
        this.error='录屏写入失败：'+error.message;this.uploadFailed=true;
        this.stop({complete:false,reason:this.error});
      }).finally(()=>{this.queued--;});
      if(this.queued>30){this.error='录屏写入过慢，已保留当前视频';this.stop({complete:false,reason:this.error});}
    };
    recorder.onerror=event=>{this.error='录屏编码失败：'+(event.error?.message||'浏览器异常');this.stop({complete:false,reason:this.error});};
    this.uploadFailed=false;
    try{recorder.start(1000);}catch(error){this.stopped=Promise.resolve();this.error=error.message;throw error;}
    this.status('recording',hasAudio?'● 正在录屏 · 含系统声音':'● 正在录屏 · 无声音',{path:this.record.path});
  }
  stop({complete=true,reason=''}={}){
    if(this.finishing)return this.finishing;
    this.version++;
    const recorder=this.recorder,stream=this.stream;this.stream=null;
    this.finishing=(async()=>{
      if(recorder){
        this.status('saving','正在保存录屏…');
        if(recorder.state!=='inactive')recorder.stop();
        stream?.getTracks().forEach(track=>track.stop());
        await this.stopped;await this.queue;
      }else stream?.getTracks().forEach(track=>track.stop());
      this.recorder=null;
      if(!this.record)return null;
      const result=await this.api('/api/recording/finish',{recording_id:this.record.id,subject_window_id:this.owner,
        chunks:this.sequence,complete:complete&&!this.error,reason:this.error||reason});
      this.status(result.complete?'saved':'partial',result.complete?'录屏已保存':'录屏已保留，完整性未确认',{...result});
      return result;
    })().catch(error=>{
      this.status('error','录屏封存失败：'+error.message+'；已写入的分段保留在本机。',{path:this.record?.path});
      return null;
    });
    return this.finishing;
  }
  get pending(){return !!this.stream||!!this.recorder;}
}
if(typeof module!=='undefined')module.exports=ScreenRecorder;
