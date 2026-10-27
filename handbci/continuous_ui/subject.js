'use strict';

const taskActions={
  thumb:{title:'拇指弯曲',file:'01_thumb.png'},
  index:{title:'食指弯曲',file:'02_index.png'},
  middle:{title:'中指弯曲',file:'03_middle.png'},
  ring:{title:'无名指弯曲',file:'04_ring.png'},
  little:{title:'小指弯曲',file:'05_little.png'},
  pinch:{title:'拇指与食指捏合',file:'06_pinch.png'}
};

// This screen describes the requested hand action, never the decoded result or imagery paradigm.
function taskPresentation(state,id){
  if(state?.subject_window_id!==id)return {title:'等待开始'};
  if(state.status==='stopping')return {title:'正在停止',detail:'正在保存本次记录'};
  if(!state.running)return {title:state.error?'已停止':state.prompted?.phase==='complete'?'本次解码完成':'已停止',detail:'记录已保存，可返回工作台查看。'};
  const p=state.prompted;
  if(!p?.enabled)return {title:'正在解码',detail:'请按已约定的任务进行。'};
  if(p.phase==='executing')return {title:'休息'};
  if(p.phase==='complete')return {title:'本次解码完成',detail:'正在保存本次记录'};
  if(['awaiting_display','cue_hold','marking_cue','decoding'].includes(p.phase)){
    const action=taskActions[p.target_action];
    return action?{title:action.title,image:'/task-images/'+action.file}:{title:'任务动作无效',error:'当前目标没有对应的手势图片。'};
  }
  return {title:'正在连接',detail:'正在检查设备并准备本次解码'};
}

function bootSubjectWindow(){
  const $=id=>document.getElementById(id),id=new URLSearchParams(location.search).get('launch');
  const token=document.querySelector('meta[name="online-token"]').content;
  let config=null,state=null,starting=false,stopping=false,offline=false,revision=0;
  let acknowledged=null,pending=null,lastRunning=false,errorMessage='',imageError='',recordingReport={},recordingSaving=false;
  const ownSession=()=>state?.running&&state.subject_window_id===id;
  async function api(path,payload){
    const abort=new AbortController(),timer=setTimeout(()=>abort.abort(),20000);
    try{
      const response=await fetch(path,{signal:abort.signal,...(payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Online-Token':token},body:JSON.stringify(payload)})});
      const data=await response.json();
      if(!response.ok)throw new Error(data.error||'本机请求失败');
      return data;
    }finally{clearTimeout(timer);}
  }
  async function uploadRecording(recordingId,owner,sequence,blob){
    const query=new URLSearchParams({recording_id:recordingId,subject_window_id:owner,sequence});
    for(let attempt=0;attempt<2;attempt++){
      const abort=new AbortController(),timer=setTimeout(()=>abort.abort(),20000);
      try{
        const response=await fetch('/api/recording/chunk?'+query,{method:'POST',signal:abort.signal,
          headers:{'Content-Type':'application/octet-stream','X-Online-Token':token},body:blob});
        const result=await response.json();if(!response.ok)throw new Error(result.error||'录屏分段保存失败');return;
      }catch(error){if(attempt===1)throw error;}
      finally{clearTimeout(timer);}
    }
  }
  const screenRecorder=new ScreenRecorder({api,upload:uploadRecording,
    capture:typeof navigator!=='undefined'&&navigator.mediaDevices?.getDisplayMedia?navigator.mediaDevices.getDisplayMedia.bind(navigator.mediaDevices):null,
    Recorder:typeof MediaRecorder!=='undefined'?MediaRecorder:null,
    onStatus:report=>{
      recordingReport=report;recordingSaving=report.phase==='saving';
      if(window.opener)window.opener.postMessage({type:'handbci:recording',id,recording:report},location.origin);
      render();
    }});
  const prompts=new ActionPrompts(new CollectionAudio(api,()=>{}),
    p=>ownSession()&&!stopping&&state.status!=='stopping'&&document.visibilityState==='visible'&&state.prompted?.session_id===p.session_id&&state.prompted?.prompt_id===p.prompt_id,
    error=>fail('动作提示音失败：'+error.message));
  const imageReady=Promise.all(Object.values(taskActions).map(action=>new Promise((resolve,reject)=>{
    const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('手势图片加载失败，请重新打开任务窗口。'));image.src='/task-images/'+action.file;
  })));
  let imagesLoaded=false;
  imageReady.then(()=>{imagesLoaded=true;render();}).catch(error=>{imageError=error.message;render();});

  function render(){
    let view=taskPresentation(state,id);
    if(offline)view={title:'连接中断',detail:'请检查解码工作台的连接状态。'};
    else if(stopping)view={title:'正在停止',detail:'正在保存本次记录'};
    else if(starting&&!ownSession())view={title:'正在准备',detail:'正在准备图片、语音与设备连接'};
    else if(!ownSession()&&!state?.subject_window_id)view={title:'等待开始',detail:config?'将此窗口移到扩展屏，再点击开始。':'请从解码工作台打开此窗口。'};
    const image=$('taskImage');
    image.hidden=!view.image;
    if(view.image&&image.getAttribute('src')!==view.image){image.src=view.image;image.alt=view.title+'目标动作示意图';}
    if(!view.image){image.removeAttribute('src');image.alt='';}
    if($('taskTitle').textContent!==view.title)$('taskTitle').textContent=view.title;
    if($('taskDetail').textContent!==(view.detail||''))$('taskDetail').textContent=view.detail||'';
    $('taskDetail').hidden=!view.detail;
    const message=errorMessage||imageError||view.error||(state?.subject_window_id===id?state.error:'');
    $('taskError').textContent=message||'';$('taskError').hidden=!message;
    $('startTask').hidden=ownSession()||starting||stopping;
    $('startTask').disabled=!config||!imagesLoaded||!!state?.running||starting||stopping||offline;
    $('startTask').disabled=$('startTask').disabled||recordingSaving;
    $('stopTask').hidden=!ownSession()&&!starting&&!stopping;
    $('stopTask').disabled=stopping||state?.status==='stopping';
    document.body.classList.toggle('is-running',!!ownSession()||starting||stopping);
    document.body.classList.toggle('has-image',!!view.image);
    $('taskRecording').hidden=!config?.record_screen&&!recordingReport.message;
    $('taskRecordingStatus').textContent=recordingReport.message||(config?.record_screen?'已选择录屏：开始时请选择主屏，需要声音时同时共享系统音频。':'');
    $('taskRecordingPath').textContent=recordingReport.path||'';
    const link=$('taskRecordingDownload');link.hidden=!recordingReport.download_url;
    if(recordingReport.download_url){link.href=recordingReport.download_url;link.setAttribute('download','');}
  }
  async function stop(){
    revision++;prompts.cancel(state?.prompted?.session_id);
    stopping=true;render();
    if(starting&&!ownSession())screenRecorder.stop({complete:false,reason:'开始前已取消'});
    try{await api('/api/subject/stop',{subject_window_id:id});}
    finally{stopping=false;render();}
  }
  function fail(message){
    errorMessage=message;
    if(ownSession()||starting)stop().catch(()=>{});
    render();
  }
  const painted=()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
  async function acknowledge(){
    const p=state?.prompted,key=p?.session_id+':'+p?.prompt_id,currentRevision=revision;
    if(!ownSession()||starting&&!prompts.prepared||stopping||state.status==='stopping'||p?.phase!=='awaiting_display'||!p.prompt_id||key===acknowledged||key===pending||document.visibilityState!=='visible')return;
    pending=key;
    const current=()=>currentRevision===revision&&ownSession()&&!stopping&&state.status!=='stopping'&&state.prompted?.prompt_id===p.prompt_id&&state.prompted?.session_id===p.session_id&&document.visibilityState==='visible';
    try{
      if(!taskActions[p.target_action])throw new Error('当前目标没有对应的手势图片。');
      await imageReady;
      await $('taskImage').decode();
      await painted();
      if(!current())return;
      const audio=await prompts.present(p);
      if(!audio||!current())return;
      await api('/api/prompted/ack',{subject_window_id:id,session_id:p.session_id,prompt_id:p.prompt_id,visible:true,client_wall_ms:Date.now(),audio});
      if(current())acknowledged=key;
    }catch(error){if(current())fail('目标提示确认失败：'+error.message);}
    finally{if(pending===key)pending=null;}
  }
  async function poll(){
    try{
      // Hidden task windows never renew the live display heartbeat.
      const query=document.visibilityState==='visible'?'?subject_window_id='+encodeURIComponent(id):'?subject_window_id=hidden';
      state=await api('/api/state'+query);offline=false;
      if(lastRunning&&!ownSession()){
        prompts.cancel();screenRecorder.stop({complete:!state.error,reason:state.error||'解码已结束'});
      }
      lastRunning=!!ownSession();render();acknowledge();
    }catch(error){offline=true;screenRecorder.stop({complete:false,reason:'本机服务连接中断'});fail('本机服务连接中断，请检查工作台。');}
    setTimeout(poll,150);
  }
  async function fullscreen(){
    if(document.fullscreenElement)await document.exitFullscreen();
    else if(document.documentElement.requestFullscreen)await document.documentElement.requestFullscreen();
  }
  $('startTask').addEventListener('click',async()=>{
    if(!config||starting||stopping||state?.running||offline||!imagesLoaded||recordingSaving)return;
    starting=true;errorMessage='';acknowledged=pending=null;const startRevision=++revision;
    // getDisplayMedia must be invoked directly from this user click, before await.
    const captureReady=config.record_screen?screenRecorder.prepare():Promise.resolve();
    const audioReady=config.prompted?.enabled?prompts.prepare():Promise.resolve();
    const fullReady=config.record_screen||document.fullscreenElement?Promise.resolve():fullscreen().catch(()=>{});
    render();
    try{
      await Promise.all([imageReady,audioReady,fullReady,captureReady]);
      if(startRevision!==revision||document.visibilityState!=='visible'){await screenRecorder.stop({complete:false,reason:'开始前已取消'});return;}
      if(config.record_screen)await screenRecorder.start(id);
      if(startRevision!==revision||document.visibilityState!=='visible'){await screenRecorder.stop({complete:false,reason:'开始前已取消'});return;}
      if(config.record_screen&&!document.fullscreenElement)await fullscreen().catch(()=>{});
      const {record_screen,...decodeConfig}=config;
      state=await api('/api/decode/start',{...decodeConfig,calibration_audio:false,subject_window_id:id});
      if(startRevision!==revision||document.visibilityState!=='visible'){
        await api('/api/subject/stop',{subject_window_id:id});await screenRecorder.stop({complete:false,reason:'开始时已取消'});return;
      }
      lastRunning=!!ownSession();render();acknowledge();
    }catch(error){
      prompts.cancel();errorMessage='开始失败：'+error.message;
      await api('/api/subject/stop',{subject_window_id:id}).catch(()=>{});
      await screenRecorder.stop({complete:false,reason:errorMessage});
    }finally{if(startRevision!==revision)await screenRecorder.stop({complete:false,reason:'开始已取消'});starting=false;render();}
  });
  $('stopTask').addEventListener('click',()=>stop().catch(error=>{errorMessage=error.message;render();}));
  $('fullscreenTask').addEventListener('click',()=>fullscreen().catch(error=>{errorMessage=error.message;render();}));
  window.addEventListener('message',event=>{
    if(event.origin!==location.origin||event.source!==window.opener||event.data?.type!=='handbci:configure'||event.data.id!==id||ownSession()||starting||stopping)return;
    config=event.data.config;errorMessage='';render();
  });
  document.addEventListener('visibilitychange',()=>{
    if(document.visibilityState!=='visible'&&(ownSession()||(starting&&recordingReport.phase!=='selecting')))fail('任务窗口已离开前台，已停止并保留记录。');
  });
  window.addEventListener('pagehide',()=>{
    prompts.cancel();revision++;
    screenRecorder.stop({complete:false,reason:'任务窗口关闭或刷新'});
    if(ownSession()||starting)fetch('/api/subject/stop',{method:'POST',keepalive:true,headers:{'Content-Type':'application/json','X-Online-Token':token},body:JSON.stringify({subject_window_id:id})}).catch(()=>{});
  });
  if(window.opener&&id)window.opener.postMessage({type:'handbci:ready',id},location.origin);
  else errorMessage='请先在解码工作台点击“打开解码任务窗口”。';
  render();poll();
}

if(typeof module!=='undefined')module.exports={taskActions,taskPresentation,bootSubjectWindow};
else bootSubjectWindow();
