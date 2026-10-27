'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="online-token"]').content;
const ns = 'http://www.w3.org/2000/svg';
const colors = ['#79dfbf','#72cbe2','#9ba9ee','#cd9bd8','#d7ba7c','#91c98d'];
let models = [], details = null, state = null, classes = [], lastCommand = 0, renderedSignature = '', busy = false, lastLogSignature = '';
let point = {x:300,y:300}, targetPoint = {x:300,y:300}, disconnected = false, offline = false;
let lastStateAt = performance.now(), selectionRevision = 0;
let pulseUntil = 0, pulseClass = null;
let workspaceKind='six', lastSixMode='acp';
let appliedDefaultsModel=null;

let discoveredDevices=[];
let discoveredTriggerDevices=[], discoveredTriggerPorts=[], triggerSearchPending=false;
let acknowledgedPrompt=null, pendingPrompt=null;
let subjectFocusPending=null,subjectFocusRevision=0,subjectFullscreenOwned=false;
let stopRequestPending=false,displayedFailure='';
let subjectWindow=null,subjectLaunch=null;
let directStartRevision=0,directOwner=null,monitorRecordingSaving=false;
let decodeLabels=[],monitorFocusActive=false,monitorFullscreenOwned=false,monitorFullscreenSession=null;
const handActions={thumb:'拇指单独动作',index:'食指单独动作',middle:'中指单独动作',ring:'无名指单独动作',little:'小指单独动作',pinch:'OK 手势'};
const decodeActionNames={thumb:'拇指',index:'食指',middle:'中指',ring:'无名指',little:'小指',pinch:'OK手势'};
const actionChannels={thumb:[4],index:[3],middle:[2],ring:[1],little:[0],pinch:[3,4,5]};
const channelNames=['小指','无名指','中指','食指','拇指弯曲','拇指旋转'];
const editable = ['sixWorkspace','fourWorkspace','mode','participant','model','host','port','eegDevice','discoverEEG','decisionMode','voteWindow','voteRatio','decisionTimeout','threshold','margin','stable','cooldown','release','outputMode','handPort','handBaud','handId','refreshPorts','checkHand','brainCalibrationMode'];
const motionLabels = {simulated:'往返模拟已完成',simulating:'往返模拟中',executing:'往返执行中',completed:'往返已到位',failed:'执行失败',cancelled:'已中止'};
editable.push('collectionMode','promptLoops','promptSeed','promptRest','promptPrepare','promptMarking','promptPort','promptWireless','promptDevice','discoverTrigger','checkTrigger','savedCalibration','recordScreen');

function text(id,value){ $(id).textContent = value; }
function element(tag, attrs={}, content){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(content!==undefined)e.textContent=content;return e;}
function option(value,label){const e=document.createElement('option');e.value=value;e.textContent=label;return e;}
function showMessage(message,error=false){const box=$(error?'errorBox':'notice');box.textContent=message;box.hidden=false;}
function clearMessages(){ $('errorBox').hidden=true; $('notice').hidden=true; }
async function api(path,payload){
  const options = payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Online-Token':token},body:JSON.stringify(payload)};
  const abort=new AbortController(),timer=setTimeout(()=>abort.abort(),path==='/api/calibration/import/finish'?300000:path==='/api/trigger/check'?40000:20000);
  options.signal=abort.signal;
  try{
    const response=await fetch(path,options);let data;
    try{data=await response.json();}catch{throw new Error('本机服务未返回有效状态。');}
    if(!response.ok)throw new Error(data.error||'操作失败');return data;
  }catch(e){if(e.name==='AbortError')throw new Error('本机请求超时；可点击停止取消连接检验。');throw e;}
  finally{clearTimeout(timer);}
}
async function uploadMonitorRecording(recordingId,owner,sequence,blob){
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
function renderRecordingReport(report){
  text('screenRecordingStatus',report.message||'');text('screenRecordingPath',report.path||'');
  $('screenRecordingResult').hidden=false;
  const link=$('screenRecordingDownload'),valid=report.download_url?.startsWith('/api/recording/file?recording_id=');
  link.hidden=!valid;if(valid){link.href=report.download_url;link.setAttribute('download','');}
  text('screenRecordingLive',report.message||'');
  $('screenRecordingLive').hidden=!['recording','saving','partial','error'].includes(report.phase);
}
const monitorRecorder=new ScreenRecorder({api,upload:uploadMonitorRecording,
  capture:typeof navigator!=='undefined'&&navigator.mediaDevices?.getDisplayMedia?navigator.mediaDevices.getDisplayMedia.bind(navigator.mediaDevices):null,
  Recorder:typeof MediaRecorder!=='undefined'?MediaRecorder:null,
  onStatus:report=>{monitorRecordingSaving=report.phase==='saving';renderRecordingReport(report);updateControls();}});
const calibrationAudio=new CollectionAudio(api,message=>text('brainAudioStatus',message));
const calibrationPrompts=new CalibrationPrompts(calibrationAudio,
  payload=>api('/api/calibration/ack',payload),
  c=>!!state?.running&&state.status!=='stopping'&&state.calibration?.session_id===c.session_id&&state.calibration?.prompt_id===c.prompt_id&&document.visibilityState==='visible',
  error=>{showMessage('语音提示失败，校准已停止：'+error.message,true);stopDecoder().catch(e=>showMessage(e.message,true));});
const actionPrompts=new ActionPrompts(new CollectionAudio(api,()=>{}),
  p=>!!state?.running&&state.status!=='stopping'&&state.prompted?.session_id===p.session_id&&state.prompted?.prompt_id===p.prompt_id&&document.visibilityState==='visible',
  error=>{showMessage('动作提示音失败，已停止：'+error.message,true);stopDecoder().catch(e=>showMessage(e.message,true));});
async function stopDecoder(){
  if(stopRequestPending)return;
  directStartRevision++;
  stopRequestPending=true;
  calibrationPrompts.cancel(state?.calibration?.session_id);
  actionPrompts.cancel(state?.prompted?.session_id);
  if(!state?.running&&monitorRecorder.pending)monitorRecorder.stop({complete:false,reason:'开始前已取消'});
  if(state?.running)state.status='stopping';
  updateControls();try{await api('/api/stop',{});}finally{stopRequestPending=false;updateControls();}
}
function at(radius,angle){const a=angle*Math.PI/180;return{x:300+Math.cos(a)*radius,y:300+Math.sin(a)*radius};}
function angleFor(index){return workspaceKind==='four'?[-135,135,-45,45][index]:-90+index*360/Math.max(2,classes.length);}
function sectorPath(angle){const half=workspaceKind==='four'?42:180/Math.max(2,classes.length)-3,a=at(238,angle-half),b=at(238,angle+half),c=at(111,angle+half),d=at(111,angle-half);return`M${a.x},${a.y} A238,238 0 0 1 ${b.x},${b.y} L${c.x},${c.y} A111,111 0 0 0 ${d.x},${d.y} Z`;}
function taskName(c){return c.task_text||c.name;}
function decodeLabel(c,action=c?.hand_action){return $('mode').value==='six'?(decodeActionNames[action]||taskName(c)):taskName(c);}
function shortName(value,max=10){return value.length>max?value.slice(0,max-1)+'…':value;}
function savedHandMapping(){
  const key=profileKey()+':mapping';let saved={};
  try{const value=JSON.parse(localStorage.getItem(key)||'{}');if(value&&typeof value==='object'&&!Array.isArray(value))saved=value;}catch{}
  const revision=details?.hand_mapping_revision;
  if(revision&&classes.some(c=>c.name==='双脚')&&classes.some(c=>c.name==='舌头')){
    let previous=null;try{previous=localStorage.getItem(key+':revision');}catch{}
    if(previous!==revision){
      for(const c of classes)if(details?.hand_mapping_reset_all||['双脚','舌头'].includes(c.name))saved[c.id]=c.hand_action;
      try{localStorage.setItem(key,JSON.stringify(saved));localStorage.setItem(key+':revision',revision);}catch{}
    }
  }
  return saved;
}
function buildDial(newClasses){
  classes=newClasses;
  decodeLabels=[];
  const savedMapping=savedHandMapping();
  $('sectors').replaceChildren();$('guides').replaceChildren();$('mapping').replaceChildren();$('actionSlots').replaceChildren();$('probabilities').replaceChildren();
  const slots=workspaceKind==='four'?4:6;
  for(let i=0;i<slots;i++){
    const angle=angleFor(i), c=classes[i], label=at(173,angle);
    const handAction=c?(state?.running?c.hand_action:savedMapping[c.id]||c.hand_action||Object.keys(handActions)[i]):null;
    const displayName=c?decodeLabel(c,handAction):'未启用';
    decodeLabels[i]=displayName;
    const group=element('g',{'data-slot':i});
    group.append(element('path',{d:sectorPath(angle),class:'sector'+(!c?' disabled':''),id:`sector${i}`}));
    group.append(element('title',{id:`sectorTitle${i}`},c?displayName:'当前模型未包含此类别'));
    group.append(element('text',{x:label.x,y:label.y-10,class:'sector-label',id:`label${i}`},shortName(displayName)));
    group.append(element('text',{x:label.x,y:label.y+10,class:'sector-prob',id:`prob${i}`},c?'—':'—'));
    if(c){
      group.append(element('line',{x1:label.x-28,y1:label.y+24,x2:label.x+28,y2:label.y+24,class:'sector-bar-track'}));
      group.append(element('line',{x1:label.x-28,y1:label.y+24,x2:label.x-28,y2:label.y+24,class:'sector-bar',id:`bar${i}`}));
      const inner=at(80,angle),outer=at(245,angle);
      $('guides').append(element('line',{x1:inner.x,y1:inner.y,x2:outer.x,y2:outer.y,class:'guide'}));
      const row=document.createElement('div');row.className='mapping-row';
      const name=document.createElement('label');name.className='field';name.textContent=`ID ${c.id} · ${c.name}\n${taskName(c)}${c.label_warning?'\n⚠ '+c.label_warning:''}`;
      const input=document.createElement('select');input.dataset.action=i;input.setAttribute('aria-label',`${taskName(c)}对应机械手动作`);
      for(const [key,label] of Object.entries(handActions))input.append(option(key,label));
      input.value=handAction||'thumb';
      input.addEventListener('change',()=>{
        decodeLabels[i]=decodeLabel(c,input.value);text(`label${i}`,decodeLabels[i]);text(`sectorTitle${i}`,decodeLabels[i]);text(`plabel${i}`,decodeLabels[i]);
        try{localStorage.setItem(profileKey()+':mapping',JSON.stringify(Object.fromEntries([...$('mapping').querySelectorAll('select')].map((e,j)=>[classes[j].id,e.value]))));}catch{}
      });
      name.append(input);row.append(name);$('mapping').append(row);
      const slot=document.createElement('span');slot.className='action-slot';slot.id=`slot${i}`;slot.title=c.action_name;$('actionSlots').append(slot);
      const pr=document.createElement('div');pr.className='prob-row';const labelText=document.createElement('span');labelText.id=`plabel${i}`;labelText.textContent=displayName;
      const track=document.createElement('div');track.className='prob-track';const fill=document.createElement('i');fill.id=`pfill${i}`;track.append(fill);
      const percent=document.createElement('b');percent.id=`pvalue${i}`;percent.textContent='—';pr.append(labelText,track,percent);$('probabilities').append(pr);
    }
    $('sectors').append(group);
  }
  text('classCount',`${classes.length} CLASS${workspaceKind!=='four'&&classes.length<6?' / 6 SLOTS':''}`);
  $('dial').setAttribute('aria-label',`${slots}类别概率、候选方向与稳定判决进度`);
  buildProfiles();

}

function requestMonitorFullscreen(){
  if(document.fullscreenElement||!document.documentElement?.requestFullscreen)return Promise.resolve();
  try{
    return Promise.resolve(document.documentElement.requestFullscreen()).then(()=>{
      monitorFullscreenOwned=true;
      if(!monitorFocusActive){monitorFullscreenOwned=false;return document.exitFullscreen?.();}
      $('fullscreenMonitor').hidden=true;
    }).catch(()=>{$('fullscreenMonitor').hidden=false;});
  }catch{return Promise.resolve();}
}
function renderMonitorFocus(){
  const active=!!state?.running&&!offline&&state.session_purpose==='decode'&&state.mode==='six'&&
    (!!state.subject_window_id||!state.prompted?.enabled);
  const wasActive=monitorFocusActive;monitorFocusActive=active;
  document.body.classList.toggle('decode-monitor-focus',active);
  $('decodeProbabilities').hidden=!active;$('monitorControls').hidden=!active;
  $('decodeWavePanel').hidden=!active;
  $('stopMonitor').disabled=state?.status==='stopping'||stopRequestPending||offline;
  $('fullscreenMonitor').hidden=!!document.fullscreenElement;
  const home=$(active?'liveProbabilityHome':'monitorProbabilityHome');
  if($('probabilities').parentNode!==home)home.append($('probabilities'));
  const waveHome=$(active?'liveWaveHome':'monitorWaveHome');
  if($('waveformContent').parentNode!==waveHome)waveHome.append($('waveformContent'));
  const sessionKey=state?.subject_window_id||state?.monitor_window_id||state?.session||'free-decode';
  if(active&&monitorFullscreenSession!==sessionKey){
    monitorFullscreenSession=sessionKey;requestMonitorFullscreen();
  }else if(!active){
    monitorFullscreenSession=null;
    if(wasActive&&monitorFullscreenOwned){
      monitorFullscreenOwned=false;
      if(document.fullscreenElement&&document.exitFullscreen)Promise.resolve(document.exitFullscreen()).catch(()=>{});
    }
  }
}
function renderProbabilities(probabilities,running,candidate){
  for(let i=0;i<(workspaceKind==='four'?4:6);i++){
    $(`sector${i}`)?.classList.toggle('active',running&&candidate===i);
    if(probabilities[i]!==undefined){
      const percent=`${(probabilities[i]*100).toFixed(1)}%`;
      text(`prob${i}`,percent);
      $(`bar${i}`)?.setAttribute('x2',Number($(`bar${i}`).getAttribute('x1'))+56*probabilities[i]);
      $(`pfill${i}`).style.width=`${100*probabilities[i]}%`;text(`pvalue${i}`,percent);
    }else if($(`pvalue${i}`)){
      text(`prob${i}`,'—');text(`pvalue${i}`,'—');$(`pfill${i}`).style.width='0%';
      $(`bar${i}`)?.setAttribute('x2',$(`bar${i}`).getAttribute('x1'));
    }
  }
}
function profileKey(){return 'hand-calibration-v2:'+(details?.hand_calibration?.shared_key||$('model').value||'none')+':'+JSON.stringify(classes.map(c=>[c.id,c.name,c.action]));}

function buildProfiles(){
  let saved={};try{saved=JSON.parse(localStorage.getItem(profileKey())||'{}');}catch{}
  const confirmed=details?.hand_calibration;
  if(confirmed?.revision&&saved.calibration_revision!==confirmed.revision)saved=confirmed;
  if(state?.running&&state.output_config?.action_profiles)saved=state.output_config;
  $('baseline').value=(saved.baseline||Array(6).fill(null)).map(v=>v??'').join(', ');
  $('handProfiles').replaceChildren();
  for(const [key,label] of Object.entries(handActions)){
    const p=saved.action_profiles?.[key]||{},row=document.createElement('div');row.className='calibration-row';row.dataset.pose=key;
    const heading=document.createElement('h3');heading.textContent=label;row.append(heading);
    const target=document.createElement('label');target.className='field';target.textContent='六路目标位置';
    const input=document.createElement('input');input.dataset.param='targets';input.value=(p.targets||Array(6).fill(null)).map(v=>v??'').join(', ');target.append(input);row.append(target);
    const fields=document.createElement('div');fields.className='pose-options';
    for(const [param,title,value,min,max] of [['speed','速度',100,1,1000],['force','力阈值 / g',300,1,3000],['timeout','单程超时 / 秒',8,.5,30],['tolerance','容差',20,1,100],['max_temperature','温度阈值 / ℃',55,1,100],['hold_seconds','目标停留 / 秒',0,0,10]]){
      const l=document.createElement('label');l.className='field';l.textContent=title;const e=document.createElement('input');e.type='number';e.dataset.param=param;e.value=p[param]??value;e.min=min;e.max=max;e.step=param==='timeout'?'.5':'1';l.append(e);fields.append(l);
    }
    const recovery=document.createElement('input');recovery.type='hidden';recovery.dataset.param='recovery';recovery.value='start';fields.append(recovery);row.append(fields);$('handProfiles').append(row);
  }
}

function positionList(value,required){if(!value.trim()&&!required)return Array(6).fill(null);const parts=value.trim().split(/[,，]/);if(parts.length!==6)throw new Error('请按固定顺序填写六路位置，用逗号分隔；未知位置留空。');return parts.map(v=>{v=v.trim();if(!v&&!required)return null;if(!/^-?\d+$/.test(v)||Number(v)<-1||Number(v)>1000)throw new Error('六路位置须为-1至1000的整数；实物输出不能留空。');return Number(v);});}
function calibrationConfig(required=false){const action_profiles={};for(const row of $('handProfiles').children){const p={};for(const e of row.querySelectorAll('[data-param]'))p[e.dataset.param]=e.dataset.param==='targets'?positionList(e.value,required):e.dataset.param==='recovery'?e.value:Number(e.value);action_profiles[row.dataset.pose]=p;}return{baseline:positionList($('baseline').value,required),action_profiles};}
function saveCalibration(){try{localStorage.setItem(profileKey(),JSON.stringify({...calibrationConfig(),calibration_revision:details?.hand_calibration?.revision||null}));}catch{}}
function outputFields(){
  $('handSettings').hidden=workspaceKind==='four';
  $('outputHint').hidden=workspaceKind==='four'||$('mode').value!=='six';
}
function outputConfig(){
  if(workspaceKind==='four')return{mode:'simulation'};
  if($('outputMode').value==='simulation')return{mode:'simulation',...calibrationConfig()};
  if(!['acp','six'].includes($('mode').value))throw new Error('演示与人工脑电只允许模拟输出。真实机械手需选择ACP真实模型。');
  return{mode:'inspire',port:$('handPort').value.trim(),baudrate:Number($('handBaud').value),hand_id:Number($('handId').value),...calibrationConfig(true)};
}


function decisionFields(){
  const vote=$('decisionMode').value==='vote';
  $('voteSettings').hidden=!vote;$('stableSettings').hidden=vote;
  const prompted=workspaceKind==='six'&&$('collectionMode').value==='prompted';
  const timing=prompted?`回位后间隔${$('cooldown').value}秒，包含休息${$('promptRest').value}秒和预提示${$('promptPrepare').value}秒，不额外叠加。首个任务也预提示${$('promptPrepare').value}秒。正式开始后再积累完整的新脑电窗口，之后才开始投票。`:`回位后先间隔${$('cooldown').value}秒，再积累完整的新脑电窗口，之后才开始投票。`;
  text('promptTimingHint',`回位后间隔共${$('cooldown').value}秒：休息${$('promptRest').value}秒＋预提示${$('promptPrepare').value}秒。两段包含在总间隔内；总间隔可在“判决参数”中修改。`);
  text('decisionHint',vote?`最近${$('voteWindow').value}秒的预测投票，达到${Math.round(Number($('voteRatio').value)*100)}%得票即触发；最长判决等待${$('decisionTimeout').value}秒，从本轮有效预测开始计算。超时后仍按最近窗口最高票选择，同票比较平均概率，再比较最新一票。${timing}滤波历史和改进模型的多个窗口也会计入等待。动作执行中不积票、不排队；缺少有效预测时不强行输出。`:'同类持续激活只触发一次；释放或换类后重新积累。阈值拒识不等于识别出休息。');
  text('pageSubtitle',vote?'汇总多次预测，让多数结果触发一次动作。':workspaceKind==='four'?'查看四类实时概率与稳定判决 · 仅网页显示':'观察每一次预测，让稳定的意图成为动作。');
}
function decisionConfig(){
  return $('decisionMode').value==='vote'?{mode:'vote',vote_window_seconds:Number($('voteWindow').value),vote_ratio:Number($('voteRatio').value),decision_timeout_seconds:Number($('decisionTimeout').value),cooldown_seconds:Number($('cooldown').value)}:{mode:'stable',confidence:Number($('threshold').value),margin:Number($('margin').value),stable_seconds:Number($('stable').value),cooldown_seconds:Number($('cooldown').value),release_seconds:Number($('release').value)};
}
function workspaceFields(){
  const four=workspaceKind==='four';document.body.classList.toggle('four-workspace',four);
  $('fourWorkspace').hidden=!models.some(m=>m.source==='dy_four_acp');
  $('promptedSettings').hidden=four;
  if(four)$('collectionMode').value='continuous';
  promptedFields();
  for(const [id,on] of [['sixWorkspace',!four],['fourWorkspace',four]]){$(id).classList.toggle('selected',on);$(id).setAttribute('aria-pressed',String(on));}
  const selectedMode=four?'four':($('mode').value&&$('mode').value!=='four'?$('mode').value:lastSixMode);
  $('mode').replaceChildren(...(four?[["four","真实脑电 · ACP直连"]]:[["acp","导入模型 · ACP直连"],["simulated","导入模型 · 模拟脑电"],["demo","界面演示（不推理）"],...(models.some(m=>m.source==='p01_six_acp_observation')?[["six","原研究模型 · 六分类"]]:[])]).map(([value,label])=>option(value,label)));
  $('mode').value=selectedMode;
  $('outputField').hidden=four;$('mappingSettings').hidden=four;
  $('calibrationPanel').hidden=four;
  text('pageTitle',four?'四分类实时解码测试':'持续解码工作台');
  text('pageSubtitle',four?'查看四类实时概率与稳定判决 · 仅网页显示':'观察每一次预测，让稳定的意图成为动作。');
  text('predictionNote',four?'每次预测均显示四路概率；满足稳定条件后再记录一次判决。':'当前预测持续变化；动作是否到位以执行状态为准。');
  text('commandLabel',four?'本次稳定判决':'本次动作指令');text('logTitle',four?'稳定判决记录':'动作记录');
  text('logNote',four?'预测不等于真实意图；无提示测试不自动计算准确率。':'实际动作以机械手回读为准');
  renderLogHeader();
  text('footerNote',four?'四分类模型测试 · 不连接机械手 · 无提示持续解码':'六类任务与动作独立映射 · 机械手按需启用');
  decisionFields();outputFields();
}
function setWorkspace(kind){
  if(state?.running||busy||workspaceKind===kind)return;
  clearMessages();
  if(kind==='four'){lastSixMode=$('mode').value;$('mode').value='four';$('outputMode').value='simulation';}
  else{$('mode').value=lastSixMode;}
  $('decisionMode').value=kind==='four'?'stable':'vote';$('cooldown').value=kind==='four'?'2':'4';
  workspaceKind=kind;point=targetPoint={x:300,y:300};pulseUntil=0;lastCommand=0;lastLogSignature='';workspaceFields();loadParticipants();
}
function matchesSource(m,mode){return mode==='six'?m.source==='p01_six_acp_observation':mode==='four'?m.source==='dy_four_acp':mode==='demo'?m.source==='probability_demo':mode==='simulated'?m.source==='synthetic_tcp':m.source==='neuracle_acp';}
function loadParticipants(){
  const mode=$('mode').value;
  $('outputMode').querySelector('option[value="inspire"]').disabled=!['six','acp'].includes(mode);
  outputFields();
  const available=models.filter(m=>matchesSource(m,mode));
  $('participant').replaceChildren();
  for(const person of [...new Set(available.map(m=>m.participant_id))])$('participant').append(option(person,person==='DEMO'?'DEMO · 界面演示':person));
  if(!available.length)$('participant').append(option('','暂无匹配被试'));
  if(state?.running&&available.some(m=>m.participant_id===state.selection?.participant_id))$('participant').value=state.selection.participant_id;
  else if(available.some(m=>m.is_default))$('participant').value=available.find(m=>m.is_default).participant_id;
  loadModels();
  $('deviceFields').hidden=!['six','four','acp'].includes(mode);

  text('deviceHint','自动搜索并连接设备，无需填写地址，也无需先在采集平台连接。其他程序若正在占用设备，请先断开该连接。');
  text('sourceHint',{six:'基础模型直接解码；改进模型可使用上次校准或重新校准。',demo:'演示概率由程序生成，无设备连接，不代表脑电识别效果。',simulated:'使用本机人工脑电，经真实接收与已训练模型推理；不是实机效果。',acp:'只接受ACP采集记录训练的已登记模型；连接在点击开始后建立。',four:'加载已有四类权重，实时显示左握拳、左旋臂、右握拳、右旋臂的概率。无需TriggerBox，不连接机械手。'}[mode]);
}
function loadModels(){
  const mode=$('mode').value;
  const available=models.filter(m=>m.participant_id===$('participant').value&&matchesSource(m,mode));
  $('model').replaceChildren();for(const m of available)$('model').append(option(m.model_id,m.name));
  if(!available.length)$('model').append(option('','尚无匹配模型'));
  if(state?.running&&available.some(m=>m.model_id===state.selection?.model_id))$('model').value=state.selection.model_id;
  else if(available.some(m=>m.is_default))$('model').value=available.find(m=>m.is_default).model_id;
  loadDetails();
}
async function loadDetails(){
  const revision=++selectionRevision;details=null;$('start').disabled=true;$('check').disabled=true;
  if(!state?.running){buildDial([]);renderedSignature='[]';text('contextText','正在核对模型与原始采集来源…');text('contextNote','尚未连接设备');}
  if(!$('model').value){text('modelMeta','暂无匹配模型。可用现有模型登记命令登记；旧8712模型不会直接用于ACP。');return;}
  try{
    const next=await api(`/api/model?participant=${encodeURIComponent($('participant').value)}&model_id=${encodeURIComponent($('model').value)}`);
    if(revision!==selectionRevision)return;
    if(next.requires_prompted_calibration&&$('collectionMode').value==='prompted')next.requires_calibration=true;
    if(next.requires_calibration&&!Object.hasOwn(next,'previous_calibration'))throw new Error('后台正在更新，暂时无法读取校准记录，请稍后刷新。');
    details=next;
    if(!state?.running&&appliedDefaultsModel!==details.model_id){
      const fields=['decisionMode','voteWindow','voteRatio','decisionTimeout','cooldown','threshold','margin','stable','release','collectionMode','promptRest','promptPrepare','promptLoops','promptMarking','outputMode'];
      for(const id of fields)if(Object.hasOwn(details.ui_defaults||{},id))$(id).value=details.ui_defaults[id];
      if(Object.hasOwn(details.ui_defaults||{},'calibrationSound'))$('calibrationSound').checked=!!details.ui_defaults.calibrationSound;
      appliedDefaultsModel=details.model_id;promptedFields();decisionFields();outputFields();
    }
    const previous=populateSavedCalibrations();
    $('brainCalibrationMode').querySelector('option[value="reuse"]').disabled=!previous;
    $('brainCalibrationMode').querySelector('option[value="reuse"]').textContent=previous?'使用已保存校准':'使用已保存校准（暂无完整记录）';
    $('brainCalibrationMode').value=state?.running?(state.selection?.calibration_mode||'new'):(previous?'reuse':'new');
    renderSavedCalibrationInfo();
    if(!state?.running)selectSavedHandConnection();
    if(!state?.running){buildDial(details.classes);renderedSignature=JSON.stringify(details.classes);}
    text('modelMeta',details.source==='p01_six_acp_observation'?(details.requires_calibration?`${details.name}\n本次连接先校准${calibrationRounds()}轮，共${calibrationTotal()}次任务。`:'基础模型 · 六分类IFNet\n连接后直接持续解码。'):details.source==='dy_four_acp'?'4类 · 实时输入32路 / 模型30路\n1000 → 250 Hz · 窗口2 s\n旧模型历史参考设置未完整记录，实时效果须实测。':details.source==='probability_demo'?'6类 · 人工概率演示 · 无脑电输入':`${details.n_classes}类 · ${details.channels.length}通道 · ${details.sample_rate} Hz\n窗口 ${details.window_seconds} s · ${details.unit}`);
    text('contextText',$('mode').value==='demo'?'六类交互演示':`${$('participant').value} / ${$('model').selectedOptions[0]?.textContent||''}`);
    text('contextNote',{six:details.requires_calibration?`改进模型 → ${calibrationTotal()}次脑电校准 → 持续解码`:'基础模型 → 持续解码',demo:'概率演示 · 无模型推理 · 无真实脑电',simulated:'人工信号 → 已训练模型 → 模拟动作',acp:'Neusen ACP → 已训练模型 → 模拟动作',four:'ACP实时采集 → 既有四类模型 → 网页识别结果'}[$('mode').value]);
    calibrationChoiceChanged();
  }catch(e){if(revision===selectionRevision)showMessage(e.message,true);}
}
function promptedFields(){
  const on=workspaceKind==='six'&&['six','acp'].includes($('mode').value)&&$('collectionMode').value==='prompted';
  $('promptedFields').hidden=!on;$('promptHardware').hidden=$('promptMarking').value!=='serial';
  $('promptWirelessFields').hidden=!$('promptWireless').checked;
}
function syncPromptTiming(source){
  if(workspaceKind!=='six'||$('collectionMode').value!=='prompted')return;
  const prepare=Number($('promptPrepare').value),rest=Number($('promptRest').value),total=Number($('cooldown').value);
  if(source==='cooldown'){
    if(total<prepare){showMessage('回位后间隔不能短于预提示准备时间。',true);decisionFields();return;}
    $('promptRest').value=Number((total-prepare).toFixed(6));
  }else $('cooldown').value=Number((rest+prepare).toFixed(6));
  decisionFields();
}
function promptedConfig(){
  const enabled=workspaceKind==='six'&&['six','acp'].includes($('mode').value)&&$('collectionMode').value==='prompted';
  const rest=Number($('promptRest').value),prepare=Number($('promptPrepare').value);
  if(enabled&&Math.abs(Number($('cooldown').value)-rest-prepare)>1e-6)throw new Error('回位后间隔须等于休息时间与预提示时间之和，请检查这三个设置。');
  return{enabled,loops:Number($('promptLoops').value),rest_seconds:rest,prepare_seconds:prepare,
  seed:$('promptSeed').value.trim()===''?null:Number($('promptSeed').value),marking:$('promptMarking').value,
  trigger_port:$('promptPort').value,trigger_host:$('promptWireless').checked?$('promptDevice').value:''};}
function requestConfig(){return{participant:$('participant').value,model_id:$('model').value,mode:$('mode').value,host:$('host').value.trim(),port:Number($('port').value),prompted:promptedConfig(),
  calibration_mode:usesPreviousCalibration()?'reuse':'new',calibration_source:usesPreviousCalibration()?selectedCalibration()?.id:undefined,
  calibration_audio:!!details?.requires_calibration&&!usesPreviousCalibration()&&$('calibrationSound').checked,output:outputConfig(),hand_actions:[...$('mapping').querySelectorAll('select')].map(e=>e.value),decision:decisionConfig()};}
function usesPreviousCalibration(){return !!details?.requires_calibration&&$('brainCalibrationMode').value==='reuse';}
function calibrationRounds(){return details?.calibration_rounds||3;}
function calibrationTotal(){return details?.calibration_trials||calibrationRounds()*6;}
function calibrationRecords(){return details?.calibration_history||(details?.previous_calibration?[details.previous_calibration]:[]);}
function selectedCalibration(){
  const records=calibrationRecords();
  return records.find(c=>c.id===$('savedCalibration').value)||(!Object.hasOwn(details||{},'calibration_history')?details?.previous_calibration:null);
}
function populateSavedCalibrations(){
  const records=calibrationRecords(),old=$('savedCalibration').value;
  let preferred=old;try{preferred=localStorage.getItem('brain-calibration-choice:'+details.participant_id+':'+details.model_id)||old;}catch{}
  if(state?.running&&state.selection?.calibration_reuse)preferred=state.selection.calibration_reuse.id;
  $('savedCalibration').replaceChildren(option('',records.length?'选择完整校准记录':'暂无完整可复用记录'));
  for(const c of records)$('savedCalibration').append(option(c.id,`${c.saved_at} · ${c.calibration_trials||calibrationTotal()}/${c.calibration_trials||calibrationTotal()}${c.external_acquisition?' · 外部采集/想象'+c.imagery_seconds+'秒':c.shared_calibration_input?' · 共用脑电/重算本模型':''} · ${c.id}`));
  for(const c of details?.unavailable_calibrations||[]){const o=option('unavailable:'+c.id,`${c.id} · 不可用：${c.reason}`);o.disabled=true;$('savedCalibration').append(o);}
  if(records.length)$('savedCalibration').value=records.some(c=>c.id===preferred)?preferred:records[0].id;
  return selectedCalibration();
}
function renderSavedCalibrationInfo(){
  const previous=selectedCalibration();
  $('savedCalibrationField').hidden=!details?.requires_calibration;
  text('brainCalibrationHistory',previous?`所选校准：${previous.saved_at} · ${calibrationTotal()} / ${calibrationTotal()} · 设备 ${previous.device.serial||'人工源'}${previous.external_acquisition?' · 外部采集：准备'+previous.prepare_seconds+'秒，想象'+previous.imagery_seconds+'秒':''}${previous.shared_calibration_input?' · 使用原校准脑电，按当前模型重算参数':''}`:`没有完整可复用的校准；完成${calibrationTotal()}次任务并保存，或从左侧导入外部校准数据。`);
  const unavailable=details?.unavailable_calibrations||[];
  text('calibrationUnavailable',unavailable.length?`${unavailable.length}条原记录不可用；最近一条：${unavailable[0].reason}`:'');
}
function calibrationChoiceChanged(){
  renderSavedCalibrationInfo();
  if(details?.requires_calibration){
    const labelled=!!details.hand_supervised_calibration;
    const title=details.name||(labelled?'六组模型 · 有标签校准（实验）':'改进模型 · IFNet＋脚舌复判');
    const timing=`${calibrationRounds()}轮，每轮${classes.length}类，共${calibrationTotal()}次；每次想象${details.calibration_imagery_seconds||4}秒，休息${details.calibration_rest_seconds||3}秒。`;
    text('brainCalibrationHint',usesPreviousCalibration()?`使用上次已完成的${calibrationTotal()}次校准，无需再执行任务。`:timing+(details.right_hand_review?'按提示想象动作，完成后启用右手复核。':labelled?'':'按提示进行运动想象。'));
    text('modelMeta',title+'\n'+(usesPreviousCalibration()?'完整校准已就绪，点击下方开始解码。':'先点击开始校准，完成后再单独开始解码。')+(labelled?'\n实验功能，在线改善尚未验证。':''));
    text('contextNote',usesPreviousCalibration()?title+' · 校准已就绪 · 等待开始解码':title+' · 先校准 · 完成后待命');
  }
  renderBrainCalibration();updateControls();
}
async function action(button,fn){if(busy)return;busy=true;button.disabled=true;clearMessages();try{await fn();}catch(e){showMessage(e.message,true);}finally{busy=false;updateControls();if(triggerSearchPending)setTimeout(searchTriggerDevices,0);}}
function updateControls(){
  const running=!!state?.running;
  const testing=['testing','cancelling'].includes(state?.eeg_connection_test?.phase)||state?.eeg_discovery?.phase==='searching'||['testing','cancelling'].includes(state?.trigger_connection_test?.phase)||state?.trigger_discovery?.phase==='searching';
  const deviceReady=!['six','four','acp'].includes($('mode').value)||!!$('host').value;
  for(const id of ['importParticipant','importName','importFile','importModel'])$(id).disabled=running||testing||busy||offline;
  for(const id of ['calibrationImportFolder','calibrationImportZip','importCalibration'])$(id).disabled=running||testing||busy||offline||!details||!['six','acp'].includes($('mode').value);
  const handReady=$('outputMode').value!=='inspire'||!!$('handPort').value;
  for(const id of editable)$(id).disabled=running||testing||busy||offline;
  for(const input of document.querySelectorAll('#mapping select, #calibrationPanel input, #calibrationPanel select, #calibrationPanel button'))input.disabled=running||testing||busy||offline;
  const triggerReady=$('collectionMode').value!=='prompted'||$('promptMarking').value!=='serial'||(!!$('promptPort').value&&(!$('promptWireless').checked||!!$('promptDevice').value));
  const calibrationReady=!details?.requires_calibration||(usesPreviousCalibration()&&!!selectedCalibration());
  const prompted=workspaceKind==='six'&&$('collectionMode').value==='prompted';
  $('start').disabled=running||testing||busy||monitorRecordingSaving||!details||!deviceReady||!handReady||!triggerReady||!calibrationReady||offline;
  $('recordingOptions').hidden=workspaceKind!=='six';
  $('start').textContent=prompted?'▶ 打开解码任务窗口':'▶ 开始解码';
  $('startCalibration').disabled=running||testing||busy||monitorRecordingSaving||!details?.requires_calibration||!deviceReady||offline;
  text('decodeStartHint',!calibrationReady?'先完成上方脑电校准，或选择使用上次完整校准。':prompted?'先打开任务窗口并移到扩展屏，在新窗口点击“开始解码”后才连接设备。本页保留实时解码显示。':workspaceKind==='six'?'自由持续解码直接在本页开始，显示圆盘、实时概率和脑电波形，不显示或播报目标提示。':'直接在本页开始四分类解码，仅网页显示。');
  text('recordingHint',prompted?'可选。任务窗口点击开始后选择显示圆盘和波形的主屏；需要语音时勾选“共享系统音频”。解码结束后自动保存到本机。':'可选。点击本页“开始解码”时选择要录制的展示屏；解码结束后自动保存到本机。');
  $('check').disabled=running||testing||busy||!details||!deviceReady||offline;
  $('checkEEG').disabled=running||testing||busy||!details||!deviceReady||!['six','four','acp'].includes($('mode').value)||offline;
  $('checkHand').disabled=running||testing||busy||!$('handPort').value||offline;
  $('checkTrigger').disabled=running||testing||busy||!details||!deviceReady||!triggerReady||$('promptMarking').value!=='serial'||offline;
  $('stop').disabled=(!running&&!testing)||state?.status==='stopping'||stopRequestPending||offline;
  $('stop').textContent=!running&&testing?'取消连接检验':'停止并保存';
  $('stopCalibration').disabled=$('stop').disabled;
  $('stopPrompted').disabled=$('stop').disabled;
  $('calibrationSound').disabled=running||busy||offline;
  $('previewCalibrationAudio').disabled=running||busy||offline;
  $('savedCalibration').disabled=running||testing||busy||offline||!usesPreviousCalibration()||!calibrationRecords().length;

}
function renderBrainCalibration(){
  const c=state?.calibration, active=!!state?.running&&c?.required, focus=active&&!c.ready;
  const selected=details?.requires_calibration;
  document.body.classList.toggle('calibration-focus',!!focus);
  $('brainCalibration').hidden=workspaceKind==='four'||!(active||selected)||(active&&c.ready);
  $('calibrationChoice').hidden=!!active;
  $('calibrationAudioSettings').hidden=!!focus||usesPreviousCalibration();
  $('stopCalibration').hidden=!focus;
  $('startCalibration').hidden=!!focus;
  const total=active?(c.total||calibrationTotal()):calibrationTotal();
  const completed=active?(c.completed||0):usesPreviousCalibration()?total:0;
  $('brainCalibrationCount').textContent=String(completed)+' / '+total;
  $('brainCalibrationProgress').style.width=completed/total*100+'%';
  const label=!active?(usesPreviousCalibration()?'使用上次校准':'准备重新校准'):state.status==='stopping'?'正在停止':c.ready?'校准完成':c.phase==='imagery'?c.task:c.phase==='rest'?'放松休息':c.stream_recovery_count?'重新准备当前任务':'准备校准';
  $('brainCalibrationTask').textContent=label;
  $('brainCalibrationDetail').textContent=!active?(usesPreviousCalibration()?'校准已就绪；到下方解码区域点击开始解码。也可点击开始校准重新采集。':`点击开始校准，按语音和文字完成${total}次任务；完成后保存并待命。`):c.phase==='rest'?'下一项：'+c.task:c.phase==='imagery'?'听到休息提示后放松。':c.ready?(state.session_purpose==='calibration'?'校准完成，正在保存；保存后等待手动开始解码。':'已载入完整校准，正在准备解码。'):'请保持放松，等待提示。';
  if(active&&!c.ready&&c.phase==='warming'&&c.stream_recovery_count)$('brainCalibrationDetail').textContent=`接收异常后的窗口已清空，已完成的${c.completed}项保留。请放松，等待当前任务重新提示。`;
  if(active&&!c.reused&&(c.awaiting_display||c.ready)&&document.visibilityState==='visible'){
    requestAnimationFrame(()=>requestAnimationFrame(()=>calibrationPrompts.present(c)));
  }
}

function subjectMode(){
  if(state?.running){
    if(state.session_purpose==='calibration'||state.calibration?.required&&!state.calibration.ready)return 'calibration';
    if(state.prompted?.enabled&&!state.subject_window_id)return 'prompted';
  }
  return subjectFocusPending;
}
function subjectPresentation(){
  const mode=subjectMode();
  if(!mode)return null;
  if(offline)return {phase:'采集提示',text:'请休息',detail:'状态更新中断'};
  if(state?.status==='stopping')return {phase:'采集提示',text:'正在停止',detail:'正在保存本次记录'};
  if(!state?.running)return {phase:mode==='calibration'?'脑电校准':'在线控制',text:'正在连接',detail:''};
  if(mode==='calibration'){
    const c=state.calibration;
    return {phase:'脑电校准',text:c?.ready?'校准完成':c?.phase==='imagery'?c.task:c?.phase==='rest'?'休息':'请保持放松',detail:''};
  }
  const p=state.prompted||{};
  if(p.phase==='executing')return {phase:'在线控制',text:'休息',detail:''};
  if(p.phase==='complete')return {phase:'在线控制',text:'本次采集完成',detail:'正在保存'};
  return {phase:'运动想象',text:p.task_text||'请保持放松',
    detail:p.target_action?'机械手：'+(handActions[p.target_action]||p.target_action):''};
}
function renderSubjectFocus(){
  const cue=subjectPresentation();
  $('subjectFocus').hidden=!cue;document.body.classList.toggle('subject-focus-active',!!cue);
  if(cue){
    text('subjectFocusPhase',cue.phase);text('subjectFocusText',cue.text);text('subjectFocusDetail',cue.detail);
    $('stopSubjectFocus').disabled=state?.status==='stopping';
  }else if(subjectFullscreenOwned){
    subjectFullscreenOwned=false;
    if(document.fullscreenElement&&document.exitFullscreen)Promise.resolve(document.exitFullscreen()).catch(()=>{});
  }
}
function requestSubjectFocus(mode){
  subjectFocusPending=mode;const revision=++subjectFocusRevision;renderSubjectFocus();
  if(!document.fullscreenElement&&document.documentElement?.requestFullscreen){
    try{Promise.resolve(document.documentElement.requestFullscreen()).then(()=>{
      if(revision===subjectFocusRevision&&subjectMode())subjectFullscreenOwned=true;
      else if(document.fullscreenElement&&document.exitFullscreen)Promise.resolve(document.exitFullscreen()).catch(()=>{});
    }).catch(()=>{});}catch{}
  }
  return revision;
}
function timeLabel(seconds){const n=Math.floor(seconds||0);return`${String(Math.floor(n/60)).padStart(2,'0')}:${String(n%60).padStart(2,'0')}`;}
function renderPrompted(){
  const p=state?.prompted,on=!!p?.enabled;
  $('promptedPanel').hidden=!on;
  if(!on){acknowledgedPrompt=pendingPrompt=null;return;}
  text('promptedCount',`${p.completed||0} / ${p.total||Number($('promptLoops').value)*6}`);
  const resting=p.phase==='executing';
  text('promptedTarget',resting?'休息':p.task_text|| (p.phase==='complete'?'本次采集完成':'等待目标提示'));
  text('promptedTask',resting||p.phase==='complete'?'':p.target_action?`机械手：${handActions[p.target_action]||p.target_action}`:'先完成设备检查与所选模型的脑电校准。');
  const labels={connecting:'正在连接设备与核验TriggerBox',waiting_calibration:'等待脑电校准完成',awaiting_display:'保持本次运动想象',marking_cue:'保持本次运动想象',cue_hold:'保持本次运动想象',decoding:'保持本次运动想象',executing:'放松休息',complete:'全部试次完成，正在保存数据'};
  text('promptedStatus',`${p.round?'第'+p.round+'轮 · ':''}${!state.running?'会话已停止，记录已保存':labels[p.phase]||p.phase}${p.marking==='software'?' · 软件估计标记':''}${state.output_mode!=='inspire'?' · 机械手模拟':''}`);
  const promptSession=p.session_id,promptId=p.prompt_id,key=promptSession+':'+promptId;
  if(!state.subject_window_id&&state.running&&state.status!=='stopping'&&p.phase==='awaiting_display'&&p.prompt_id&&key!==acknowledgedPrompt&&key!==pendingPrompt&&document.visibilityState==='visible'){
    pendingPrompt=key;
    requestAnimationFrame(()=>requestAnimationFrame(async()=>{
      if(!state?.running||state.status==='stopping'||state.prompted?.session_id!==promptSession||state.prompted?.prompt_id!==promptId||document.visibilityState!=='visible'){if(pendingPrompt===key)pendingPrompt=null;return;}
      try{
        const audio=await actionPrompts.present(p);
        if(!audio||!state?.running||state.status==='stopping'||state.prompted?.session_id!==promptSession||state.prompted?.prompt_id!==promptId)return;
        await api('/api/prompted/ack',{session_id:promptSession,prompt_id:promptId,visible:true,client_wall_ms:Date.now(),...(audio?{audio}:{})});
        if(state?.prompted?.session_id===promptSession&&state.prompted?.prompt_id===promptId)acknowledgedPrompt=key;
      }
      catch(e){if(state?.running&&state.prompted?.session_id===promptSession&&state.prompted?.prompt_id===promptId){showMessage('目标提示确认失败，已停止：'+e.message,true);await stopDecoder();}}
      finally{if(pendingPrompt===key)pendingPrompt=null;}
    }));
  }
}
function update(s){
  if(!s.running&&(s.mode!==$('mode').value||(s.selection&&s.selection.model_id!==$('model').value))){
    s={...s,classes:classes,mode:$('mode').value,probabilities:[],commands:[],command_count:0,n_predictions:0,wall_seconds:0,
       samples_received:0,source_seconds:0,rate_hz:null,inference_ms:null,gap_count:0,calibration:null,expected_rate:details?.sample_rate,
       gate:{candidate:null,progress:0,reason:'已选择新配置，等待开始'}};
  }
  const wasRunning=state?.running,previousSession=state?.calibration?.session_id,previousPromptSession=state?.prompted?.session_id;state=s;if(wasRunning&&!s.running){calibrationPrompts.cancel(previousSession);actionPrompts.cancel(previousPromptSession);setTimeout(loadDetails,0);}renderBrainCalibration();
  if(wasRunning&&!s.running&&monitorRecorder.pending)monitorRecorder.stop({complete:!s.error,reason:s.error||'解码已结束'});
  renderPrompted();renderSubjectFocus();
  const signature=JSON.stringify(s.classes);
  if(s.running&&signature!==renderedSignature){buildDial(s.classes);renderedSignature=signature;}
  renderMonitorFocus();
  const status={idle:'待机',starting:'正在准备',running:s.mode==='demo'?'演示运行中':'持续解码中',stopping:'正在保存',stopped:'已停止',failed:'运行异常'}[s.status]||s.status;
  text('statusBadge',`● ${s.running&&s.phase==='calibrating'?'脑电校准中':status}`);$('statusBadge').className=`status-badge ${s.running?'running':''} ${s.status==='failed'?'failed':''}`;
  const failureKey=s.error?`${s.session||''}:${s.error}`:'';
  if(failureKey&&failureKey!==displayedFailure&&!busy){$('notice').hidden=true;showMessage(s.error,true);displayedFailure=failureKey;}
  const four=workspaceKind==='four',real=!four&&(s.running?s.output_mode:$('outputMode').value)==='inspire', hand=s.hand||{};
  text('outputTitle',four?'稳定识别结果':real?'机械手动作':'模拟动作');text('outputBadge',four?'4 CLASS / LIVE':real?'INSPIRE / REAL':'SIMULATED');
  const realHandState=!s.running&&!hand.connected&&!hand.busy&&hand.phase!=='failed'?'待开始 · 开始后先伸直五指':({disconnected:'机械手未连接',initializing:'初始归位 · 等待五指伸直',ready:hand.initialization?.execution_status==='completed'?'五指伸直就绪':'机械手已连接',executing:'动作去程 · 等待到位',holding:'目标到位 · 正在停留',recovering:'返回本次起始位置 · 等待到位',completed:'一次往返已完成',failed:'动作失败 · 已停止',cancelled:'动作已中止'}[hand.phase]||'等待机械手状态');
  text('handState',four?'仅网页显示 · 不连接机械手':real?realHandState:hand.busy?hand.phase==='recovering'?'网页模拟：正在返回 · 未连接机械手':hand.phase==='holding'?'网页模拟：目标停留 · 未连接机械手':'网页模拟：动作去程 · 未连接机械手':'未连接机械手 · 模拟输出');
  text('motionDuration',hand.last_duration_seconds===null||hand.last_duration_seconds===undefined?'—':`${hand.last_duration_seconds.toFixed(2)} s`);
  if(!s.error&&hand.error&&hand.phase==='failed'&&displayedFailure!==`hand:${hand.error}`&&!busy){showMessage(`机械手：${hand.error}`,true);displayedFailure=`hand:${hand.error}`;}
  const p=signature===renderedSignature?(s.probabilities||[]):[], max=p.length?Math.max(...p):null, top=max===null?null:p.indexOf(max);
  const candidate=s.gate?.candidate, progress=s.gate?.progress||0;
  renderProbabilities(p,s.running,candidate);
  const active=s.running&&candidate!==null&&candidate!==undefined&&progress>0;
  targetPoint=active?at(progress*230,angleFor(candidate)):{x:300,y:300};
  text('hubTitle',s.running?(candidate!==null&&candidate!==undefined?shortName(decodeLabels[candidate]||decodeLabel(classes[candidate]),8):'暂不输出'):s.status==='idle'?'等待开始':'已停止');
  text('hubSub',s.running?(active?(s.gate?.mode==='vote'?'多次预测投票':'稳定证据积累'):s.phase==='warming'?'等待模型窗口':'等待判决结果'):'暂不输出');
  if(s.running&&s.phase==='calibrating'){text('hubTitle','脑电校准中');text('hubSub','按上方提示进行运动想象');}
  text('decisionReason',s.gate?.reason||'等待开始持续解码');text('progressValue',active?`${Math.round(progress*100)}%${s.gate?.mode==='vote'?' 得票':''}`:'—');
  text('predictedClass',top!==null?decodeLabels[top]||decodeLabel(classes[top]):'—');text('topProbability',max===null?'—':(max*100).toFixed(1));text('predictionLive',s.phase==='calibrating'?'校准中 · 暂无预测':s.running?'● 连续更新':'等待输入');
  text('wallTime',timeLabel(s.wall_seconds));text('predictionCount',String(s.n_predictions));
  text('inputRate',s.rate_hz===null||s.rate_hz===undefined?'—':`${Math.round(s.rate_hz)} Hz`);
  text('rateNote',s.mode==='demo'?'演示模式无脑电输入':`标称 ${s.expected_rate??details?.sample_rate??'—'} Hz · 收到 ${s.samples_received} 点`);
  text('inferenceNote',s.mode==='demo'?'演示概率 · 非模型推理':s.inference_ms==null?'等待模型窗口':`单次处理 ${s.inference_ms.toFixed(1)} ms`);
  text('gapCount',s.mode==='demo'?'—':String(s.gap_count));text('gapNote',s.mode==='demo'?'真实数据模式监测':s.stale_seconds>.4?`最近 ${s.stale_seconds.toFixed(1)} s 无新数据`:`数据时长 ${(s.source_seconds||0).toFixed(1)} s`);
  const count=s.command_count||0; text('commandCount',String(count).padStart(2,'0'));
  const latest=s.commands?.at(-1);
  const handAction=hand.busy&&['executing','holding'].includes(hand.phase)?latest?.hand_action:null;for(const key of Object.keys(handActions))$('handDrawing').classList.toggle('pose-'+key,!four&&handAction===key);
  const paths={Thumb:['M98 142 Q63 136 62 106 Q62 92 73 100 L106 127','M98 142 Q79 122 97 113 Q111 108 111 127'],Index:['M100 108 L100 57 Q100 40 113 45 Q120 47 119 60 L119 107','M100 108 L100 87 Q111 72 120 88 L119 107'],Middle:['M123 106 L123 34 Q123 20 136 23 Q145 26 143 41 L143 106','M123 106 L123 87 Q134 72 143 88 L143 106'],Ring:['M146 106 L146 48 Q146 34 158 38 Q167 41 165 53 L165 107','M146 106 L146 87 Q157 72 165 88 L165 107'],Little:['M168 108 L168 72 Q168 57 179 63 Q187 67 184 80 L182 111','M168 108 L168 94 Q176 81 183 95 L182 111']};
  for(const [name,pair] of Object.entries(paths)){const active=!four&&(handAction===name.toLowerCase()||handAction==='pinch'&&['Thumb','Index'].includes(name));$('finger'+name).setAttribute('d',pair[active?1:0]);}
  if(!four&&handAction==='pinch'){$('fingerThumb').setAttribute('d','M98 142 Q65 124 84 100 Q98 93 106 111 L107 127');$('fingerIndex').setAttribute('d','M100 108 Q73 91 84 76 Q111 67 119 92 L119 107');}
  if(latest){text('lastAction',latest.action_name);text('lastActionMeta',`${motionLabels[latest.execution_status]||'网页模拟'} · ${timeLabel(latest.time_seconds)} · ${latest.decision_mode==='vote'?`${Math.round(latest.vote_share*100)}%得票${latest.decision_reason==='deadline_plurality'?' · 最长等待后选择':''}`:`${(latest.confidence*100).toFixed(1)}%`}`);for(let i=0;i<classes.length;i++)$(`slot${i}`)?.classList.toggle('active',latest.class_id===i);}
  else{text('lastAction',four?'等待稳定识别':'尚未输出动作');text('lastActionMeta','稳定判决后在这里显示');for(let i=0;i<classes.length;i++)$(`slot${i}`)?.classList.remove('active');}
  if(count!==lastCommand){
    if(count>lastCommand){$('actionPanel').classList.remove('triggered');void $('actionPanel').offsetWidth;$('actionPanel').classList.add('triggered');pulseClass=latest.class_id;pulseUntil=performance.now()+280;point=at(238,angleFor(pulseClass));}
    lastCommand=count;
  }
  const logSignature=JSON.stringify([!!s.running,s.commands||[]]);if(logSignature!==lastLogSignature){renderLog(s.commands||[]);lastLogSignature=logSignature;}
  $('saved').hidden=!s.saved_path;if(s.saved_path)$('savedPath').value=s.saved_path;
  renderWaveInfo();
  if(wasRunning&&!s.running&&s.status==='stopped')showMessage(s.session_purpose==='calibration'?(s.calibration?.ready?'校准完成并已保存。点击下方“开始解码”才开始控制。':'校准已停止，未完成记录已保存。'):'解码已停止，本次脑电和动作记录已保存。');
  updateControls();if(monitorFocusActive||$('monitor').open)drawWave();
}
function actionComparison(c){
  const f=c.actual_feedback,expected=c.target_hand_action||c.hand_action;
  const instruction=handActions[expected]||c.action_name||'未记录';
  if(c.physical_action!==true)return {instruction,actual:'模拟输出',result:'模拟',tone:'neutral'};
  const confirmed=f?.source==='joint_position_readback'&&f.target_confirmed===true;
  const actual=confirmed?(f.action_name||'目标已到位'):c.execution_status==='executing'?'执行中':'未确认';
  if(c.execution_status==='failed')return {instruction,actual,result:'失败',tone:'bad'};
  if(c.execution_status==='cancelled')return {instruction,actual,result:'已中止',tone:'bad'};
  if(c.execution_status==='executing')return {instruction,actual,result:'执行中',tone:'pending'};
  if(confirmed&&f.verified===true&&c.execution_status==='completed'&&f.actual_hand_action&&expected){
    const matches=f.actual_hand_action===expected;
    return {instruction,actual,result:matches?'正确':'不一致',tone:matches?'good':'bad'};
  }
  return {instruction,actual,result:'未确认',tone:'pending'};
}
function actionScore(c,side){
  if(side==='instruction'){
    if(c.instruction_evidence)return c.instruction_evidence;
    const same=c.target_label===undefined||c.target_label===c.model_class_id;
    return {task:c.target_task||(same?c.task:null),vote_share:same?c.vote_share:null,probability:same?c.confidence:null};
  }
  const f=c.actual_feedback;
  if(c.physical_action!==true||f?.source!=='joint_position_readback'||f.target_confirmed!==true)return null;
  if(c.actual_evidence)return c.actual_evidence;
  if(f.actual_hand_action&&f.actual_hand_action===c.hand_action)return {task:c.task,vote_share:c.vote_share,probability:c.confidence};
  return null;
}
function actionWithTask(name,score){return score?.task?`${name}（${score.task}）`:name;}
function actionScoreText(score){
  const vote=Number.isFinite(score?.vote_share)?`${Math.round(score.vote_share*100)}%`:'—';
  const probability=Number.isFinite(score?.probability)?`${(score.probability*100).toFixed(1)}%`:'—';
  return `得票率 ${vote} · 概率 ${probability}`;
}
function renderLogHeader(){
  const headers=workspaceKind==='four'?['序号','运行时间','识别任务','稳定类别','预测概率 / 得票','输出方式','状态']:['序号','时间','指令动作','实际动作','结果'];
  if(workspaceKind!=='four'&&state?.running)headers.pop();
  $('logHead').replaceChildren(...headers.map(label=>{const th=document.createElement('th');th.textContent=label;return th;}));
}
function renderLog(commands){
  $('logRows').replaceChildren();
  const four=workspaceKind==='four';
  renderLogHeader();
  if(!commands.length){const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=four?7:state?.running?4:5;cell.className='empty';cell.textContent=four?'暂无稳定判决记录':'暂无动作记录';row.append(cell);$('logRows').append(row);return;}
  for(const c of [...commands].reverse()){
    const row=document.createElement('tr');
    const add=(value,className='')=>{const cell=document.createElement('td');cell.textContent=value;cell.className=className;row.append(cell);return cell;};
    add(String(c.sequence).padStart(2,'0'));add(timeLabel(c.time_seconds));
    if(!four){
      const comparison=actionComparison(c);
      for(const side of ['instruction','actual']){
        const score=actionScore(c,side),cell=add(actionWithTask(comparison[side],score),'log-action');
        const values=document.createElement('span');values.className='action-scores';values.textContent=actionScoreText(score);
        values.title='同一次判决时，对应运动想象类别的分数';cell.append(values);
      }
      if(!state?.running){
        const cell=add(''),result=document.createElement('span');result.className='result-pill feedback-'+comparison.tone;result.textContent=comparison.result;
        cell.append(result);cell.title=c.error||'';
      }
      $('logRows').append(row);continue;
    }
    add(c.task);add(c.action_name);
    add(c.decision_mode==='vote'?`${Math.round(c.vote_share*100)}%得票${c.decision_reason==='deadline_plurality'?' · 最长等待':''}`:`${(c.confidence*100).toFixed(1)}%`,'log-confidence');
    const cell=add(''),pill=document.createElement('span');pill.className='output-pill';pill.textContent='仅网页显示';cell.append(pill);add('已记录');
    $('logRows').append(row);
  }
}
function renderWaveInfo(){
  const s=state||{},p=s.prompted||{},hand=s.hand||{},wf=s.waveform;
  let phase='等待开始';
  if(offline)phase='连接中断';
  else if(s.status==='stopping')phase='正在保存';
  else if(!s.running)phase=s.error?'运行异常':s.status==='idle'?'等待开始':'已停止';
  else if(s.phase==='calibrating')phase='脑电校准';
  else if(hand.busy)phase=hand.phase==='recovering'?'回位':hand.phase==='holding'?'动作保持':'执行动作';
  else if(p.enabled){
    phase=({connecting:'设备准备',waiting_calibration:'等待校准',resting:'休息',
      awaiting_display:'准备',marking_prepare:'准备',cue_hold:'准备',marking_cue:'准备',
      decoding:'解码',executing:'休息',complete:'采集完成'})[p.phase]||'准备';
  }else phase=s.phase==='warming'?'积累脑电窗口':'解码';
  text('wavePhase',phase);
  const band=wf?.bandpass_hz,filter=band?`显示滤波 ${band[0]}–${band[1]} Hz${wf.notch_hz?' · '+wf.notch_hz+' Hz陷波':''}`:'等待信号';
  text('waveHint',s.mode==='demo'?'交互演示不生成或伪装脑电波形。':`C3 · Cz · C4 · 最近5秒 · ${filter}`);
}
function drawWave(){
  const canvas=$('wave'),w=canvas.clientWidth,h=canvas.clientHeight,dpr=window.devicePixelRatio||1;
  if(!w||!h)return;
  const ctx=canvas.getContext('2d');if(!ctx)return;
  canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
  const scale=Number($('waveScale').value)||50,wf=state?.waveform||{};
  const trace=state?.mode==='demo'?[]:wf.points||[],end=trace.at(-1)?.[0]??0;
  const points=trace.filter(p=>p[0]>=end-5),left=64,right=Math.max(left+1,w-62),bottom=h-21,rowHeight=bottom/3;
  ctx.font='11px Consolas, monospace';ctx.lineWidth=1;
  for(let i=0;i<=5;i++){
    const x=left+(right-left)*i/5;ctx.strokeStyle='#1c333f';ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,bottom);ctx.stroke();
    ctx.fillStyle='#8ca4af';ctx.textAlign='center';ctx.fillText(i===5?'0 s':`${i-5} s`,x,h-6);
  }
  ['C3','Cz','C4'].forEach((name,row)=>{
    const center=rowHeight*(row+.5),half=Math.max(1,rowHeight/2-5),source=(wf.channels||[]).indexOf(name);
    ctx.strokeStyle='#29414a';ctx.beginPath();ctx.moveTo(left,center);ctx.lineTo(right,center);ctx.stroke();
    const color=['#79dfbf','#72cbe2','#b1bdf4'][row];
    ctx.textAlign='left';ctx.fillStyle=color;ctx.font='600 15px sans-serif';ctx.fillText(name,10,center+5);
    ctx.fillStyle='#78959e';ctx.font='10px Consolas, monospace';ctx.fillText(`+${scale}`,left+3,center-half+9);ctx.fillText(`−${scale}`,left+3,center+half-2);
    if(!wf.available?.includes(name)||source<0||points.length<2){
      ctx.fillStyle='#78959e';ctx.font='12px sans-serif';ctx.fillText(wf.available?.includes(name)?'等待信号':'未接入',left+56,center+4);return;
    }
    let drawing=false,clipped=false;ctx.strokeStyle=color;ctx.lineWidth=1.3;ctx.beginPath();
    for(const p of points){
      const value=p[source+1];if(value===null||!Number.isFinite(value)){drawing=false;continue;}
      clipped ||= Math.abs(value)>scale;
      const x=left+(p[0]-(end-5))/5*(right-left),y=center-Math.max(-scale,Math.min(scale,value))/scale*half;
      if(!drawing)ctx.moveTo(x,y);else ctx.lineTo(x,y);drawing=true;
    }
    ctx.stroke();ctx.lineWidth=1;
    if(clipped){ctx.fillStyle='#f0b38b';ctx.font='11px sans-serif';ctx.fillText('超量程',right+8,center+4);}
  });
}
function animate(){
  const stale=performance.now()-lastStateAt>1200;
  if(stale){targetPoint={x:300,y:300};if(state?.running){text('decisionReason','页面状态更新中断 · 等待恢复');text('hubTitle','状态中断');text('hubSub','等待服务恢复');}}
  point.x+=(targetPoint.x-point.x)*.13;point.y+=(targetPoint.y-point.y)*.13;
  if(!stale&&state?.running&&performance.now()<pulseUntil&&pulseClass!==null)point=at(238,angleFor(pulseClass));
  for(const id of ['ball','ballGlow']){$(id).setAttribute('cx',point.x);$(id).setAttribute('cy',point.y);}
  requestAnimationFrame(animate);
}
async function poll(){
  try{const s=await api('/api/state');lastStateAt=performance.now();offline=false;if(disconnected){disconnected=false;$('errorBox').hidden=true;}update(s);}
  catch(e){offline=true;monitorRecorder.stop({complete:false,reason:'本机服务连接中断'});calibrationPrompts.cancel(state?.calibration?.session_id);actionPrompts.cancel(state?.prompted?.session_id);updateControls();renderSubjectFocus();renderMonitorFocus();if(!disconnected){showMessage('本机服务连接中断。圆盘已停止更新；后台超过15秒收不到网页心跳将停止并保存。',true);disconnected=true;}text('statusBadge','● 服务离线');}
  setTimeout(poll,150);
}
$('mode').addEventListener('change',()=>{clearMessages();if(!['six','acp'].includes($('mode').value))$('collectionMode').value='continuous';promptedFields();loadParticipants();});$('participant').addEventListener('change',loadModels);$('model').addEventListener('change',loadDetails);
$('sixWorkspace').addEventListener('click',()=>setWorkspace('six'));$('fourWorkspace').addEventListener('click',()=>setWorkspace('four'));
$('outputMode').addEventListener('change',()=>{outputFields();updateControls();});
$('brainCalibrationMode').addEventListener('change',calibrationChoiceChanged);
$('savedCalibration').addEventListener('change',()=>{const selected=selectedCalibration();if(selected){try{localStorage.setItem('brain-calibration-choice:'+details.participant_id+':'+details.model_id,selected.id);}catch{}}calibrationChoiceChanged();});
$('calibrationPanel').addEventListener('input',saveCalibration);
$('calibrationPanel').addEventListener('change',saveCalibration);
function selectSavedHandConnection(){
  const saved=details?.hand_connection;
  if($('handPort').value||!saved||![...$('handPort').options].some(p=>p.value===saved.port))return;
  $('handPort').value=saved.port;$('handBaud').value=saved.baudrate;$('handId').value=saved.hand_id;
}
async function refreshHandPorts(){
  let previous=$('handPort').value;
  if(!previous){try{previous=localStorage.getItem('hand-device-port-v1')||'';}catch{}}
  const ports=(await api('/api/hand/ports')).filter(p=>!/^BTHENUM/i.test(p.hwid||'')&&!/蓝牙|bluetooth/i.test(p.description||''));
  $('handPort').replaceChildren(option('',ports.length?'请选择机械手端口':'未发现串口'));
  for(const p of ports)$('handPort').append(option(p.port,`${p.port} · ${p.description}`));
  if(ports.some(p=>p.port===previous))$('handPort').value=previous;
  else if(ports.length===1)$('handPort').value=ports[0].port;
  selectSavedHandConnection();
  text('handCheckResult',ports.length?`发现 ${ports.length} 个串口，选择机械手对应端口后测试连接。`:'未发现串口，请检查USB连接后重新搜索。');
  updateControls();
}
$('refreshPorts').addEventListener('click',()=>action($('refreshPorts'),refreshHandPorts));
$('handPort').addEventListener('change',()=>{text('handCheckResult','端口已切换，请测试连接。');updateControls();});
function updateTriggerSelections(){
  const previous=$('promptPort').value;
  const ports=discoveredTriggerPorts.filter(p=>$('outputMode').value!=='inspire'||p.port!==$('handPort').value);
  $('promptPort').replaceChildren(option('',ports.length?'请选择TriggerBox端口':'未发现可用串口'));
  for(const p of ports)$('promptPort').append(option(p.port,`${p.port} · ${p.description}`));
  if(ports.some(p=>p.port===previous))$('promptPort').value=previous;
  else if(ports.length===1)$('promptPort').value=ports[0].port;
  text('triggerCheckResult','设备选择已更新，请检验连接与事件回传。');
  updateControls();
}
async function searchTriggerDevices(){
  if(state?.running||workspaceKind!=='six'||$('collectionMode').value!=='prompted'||$('promptMarking').value!=='serial'){triggerSearchPending=false;return;}
  if(busy){triggerSearchPending=true;return;}
  triggerSearchPending=false;
  return action($('discoverTrigger'),async()=>{
    text('triggerDiscoveryResult','正在搜索串口与同网段TriggerBox…');
    const oldPort=$('promptPort').value,oldDevice=$('promptDevice').value;
    $('promptPort').replaceChildren(option('','正在搜索…'));$('promptDevice').replaceChildren(option('','正在搜索…'));
    try{
      let saved={};try{saved=JSON.parse(localStorage.getItem('trigger-device-choice-v1')||'{}');}catch{}
      const r=await api('/api/trigger/discover',{known_hosts:saved.host?[saved.host]:[]});
      discoveredTriggerDevices=r.devices;discoveredTriggerPorts=r.ports;
      updateTriggerSelections();
      const preferredPort=oldPort||saved.port||'';
      if(preferredPort&&[...$('promptPort').options].some(p=>p.value===preferredPort))$('promptPort').value=preferredPort;
      $('promptDevice').replaceChildren(option('',r.devices.length?'请选择TriggerBox':'未发现TriggerBox'));
      for(const d of r.devices)$('promptDevice').append(option(d.ip,`${d.type} · ${d.serial||''} · ${d.ip}`));
      if(r.devices.length===1)$('promptDevice').value=r.devices[0].ip;
      else if(r.devices.some(d=>d.ip===(oldDevice||saved.host)))$('promptDevice').value=oldDevice||saved.host;
      text('triggerDiscoveryResult',r.message+(r.devices.length>1?' 请从列表选择。':!r.devices.length&&$('promptWireless').checked?' 请检查TriggerBox供电和局域网连接后重新搜索。':''));
    }catch(e){discoveredTriggerDevices=[];discoveredTriggerPorts=[];$('promptPort').replaceChildren(option('','搜索失败'));$('promptDevice').replaceChildren(option('','搜索失败'));text('triggerDiscoveryResult','搜索失败：'+e.message);throw e;}
  });
}
$('discoverTrigger').addEventListener('click',searchTriggerDevices);
for(const id of ['promptPort','promptDevice','promptWireless'])$(id).addEventListener('change',()=>{promptedFields();text('triggerCheckResult','设备选择已更新，请重新检验连接与回传。');updateControls();});
for(const id of ['outputMode','handPort'])$(id).addEventListener('change',()=>{if(discoveredTriggerPorts.length)updateTriggerSelections();});
$('checkTrigger').addEventListener('click',()=>action($('checkTrigger'),async()=>{
  text('triggerCheckResult','正在临时连接脑电与TriggerBox、同步时钟并核验事件回传…');
  const request={participant:$('participant').value,model_id:$('model').value,mode:$('mode').value,host:$('host').value,port:Number($('port').value),
    decision:decisionConfig(),prompted:promptedConfig(),hand_actions:[...$('mapping').querySelectorAll('select')].map(e=>e.value),
    output:{mode:$('outputMode').value,port:$('handPort').value}};
  try{const r=await api('/api/trigger/check',request);
    text('triggerCheckResult',`${r.message} ${r.trigger?.name||''} · ${r.test_event?.verification||''}`);
    try{localStorage.setItem('trigger-device-choice-v1',JSON.stringify({port:$('promptPort').value,host:$('promptDevice').value}));}catch{}
  }catch(e){text('triggerCheckResult','连接与回传检验失败：'+e.message);throw e;}
}));
$('checkHand').addEventListener('click',()=>action($('checkHand'),async()=>{
  text('handCheckResult','正在读取机械手状态…');
  try{
    const result=await api('/api/hand/check',{port:$('handPort').value.trim(),baudrate:Number($('handBaud').value),hand_id:Number($('handId').value)});
    try{localStorage.setItem('hand-device-port-v1',$('handPort').value);}catch{}
    text('handCheckResult',`${result.message} 位置：${result.status.angles.join(', ')}；故障码：${result.status.errors.join(', ')}。`);
  }catch(e){text('handCheckResult','连接检查失败：'+e.message);throw e;}
}));
function selectEegDevice(){
  const d=discoveredDevices.find(d=>d.ip===$('eegDevice').value);
  $('host').value=d?.ip||'';$('port').value=d?.port||4097;
  text('eegCheckResult',d?'设备已选中，点击测试脑电连接或直接开始解码。':'请选择找到的脑电设备。');
  if(d){try{localStorage.setItem('eeg-device-endpoint-v1',JSON.stringify({host:d.ip,port:d.port}));}catch{}}
  updateControls();
}
async function searchEegDevices(){
  return action($('discoverEEG'),async()=>{
    text('eegDiscoveryResult','正在自动搜索同网段脑电设备…');
    $('eegDevice').replaceChildren(option('','正在搜索…'));$('host').value='';$('port').value=4097;
    try{
      let preferred='';try{preferred=JSON.parse(localStorage.getItem('eeg-device-endpoint-v1')||'{}').host||'';}catch{}
      const r=await api('/api/eeg/discover',{known_hosts:preferred?[preferred]:[]});discoveredDevices=r.devices;
      $('eegDevice').replaceChildren(option('',r.devices.length?'请选择脑电设备':'未发现设备'));
      for(const d of r.devices)$('eegDevice').append(option(d.ip,(d.type||'脑电设备')+' · '+(d.serial||'')+' · '+d.ip));
      if(r.devices.length===1)$('eegDevice').value=r.devices[0].ip;
      else if(r.devices.some(d=>d.ip===preferred))$('eegDevice').value=preferred;
      text('eegDiscoveryResult',r.message+(r.devices.length>1?' 请核对并选择要使用的设备。':''));selectEegDevice();
    }catch(e){discoveredDevices=[];$('eegDevice').replaceChildren(option('','搜索失败'));text('eegDiscoveryResult','设备搜索失败：'+e.message);throw e;}
  });
}
$('discoverEEG').addEventListener('click',searchEegDevices);
$('eegDevice').addEventListener('change',selectEegDevice);
$('checkEEG').addEventListener('click',()=>action($('checkEEG'),async()=>{
  text('eegCheckResult','正在连接设备并检查约2秒的连续脑电数据…');
  try{
    const r=await api('/api/eeg/check',requestConfig());
    const info=r.device?.info||{};
    text('eegCheckResult',r.message+' '+(info.type||info.type_code||info.model||'脑电设备')+' · '+r.channels+'通道 · 标称'+r.sample_rate+' Hz · 实际约'+r.received_rate_hz+' Hz · 收到'+r.samples_received+'点。');
  }catch(e){text('eegCheckResult','连接测试失败：'+e.message);throw e;}
}));
for(const id of ['model','participant'])$(id).addEventListener('change',()=>text('eegCheckResult','已切换模型，请重新测试设备与模型的匹配。'));
$('check').addEventListener('click',()=>action($('check'),async()=>{const r=await api('/api/check',requestConfig());showMessage(r.message);}));
function openSubjectWindow(config){
  if(subjectWindow&&!subjectWindow.closed){
    subjectLaunch.config=config;
    subjectWindow.postMessage({type:'handbci:configure',id:subjectLaunch.id,config},location.origin);
    subjectWindow.focus();
  }else{
    const id=crypto.randomUUID();subjectLaunch={id,config};
    subjectWindow=window.open('/subject?launch='+encodeURIComponent(id),'handbci-task-'+id,'popup,width=1000,height=820,resizable=yes,scrollbars=yes');
    if(!subjectWindow){subjectLaunch=null;throw new Error('浏览器拦截了任务窗口，请允许本机页面弹出窗口后重试。');}
  }
  showMessage('任务窗口已打开，尚未连接设备。移到扩展屏后，在该窗口点击“开始解码”；更改设置后请再次点击本页开始按钮。');
}
window.addEventListener('message',event=>{
  if(event.origin!==location.origin||event.source!==subjectWindow||event.data?.id!==subjectLaunch?.id)return;
  if(event.data.type==='handbci:ready')subjectWindow.postMessage({type:'handbci:configure',id:subjectLaunch.id,config:subjectLaunch.config},location.origin);
  if(event.data.type==='handbci:recording'){
    renderRecordingReport(event.data.recording||{});
  }
});
async function startDirectDecoding(config){
  if(state?.running||monitorRecordingSaving)throw new Error('请先等待当前会话或录屏保存结束。');
  const revision=++directStartRevision,owner=crypto.randomUUID();directOwner=owner;
  // Invoke capture in this click's call stack, before waiting for any request.
  const captureReady=config.record_screen?monitorRecorder.prepare():Promise.resolve();
  calibrationPrompts.cancel();actionPrompts.cancel();
  if(subjectWindow&&!subjectWindow.closed)subjectWindow.close?.();
  subjectWindow=subjectLaunch=null;
  try{
    await captureReady;
    if(revision!==directStartRevision)throw new DOMException('开始已取消','AbortError');
    if(config.record_screen)await monitorRecorder.start(owner);
    if(revision!==directStartRevision)throw new DOMException('开始已取消','AbortError');
    const {record_screen,...decodeConfig}=config;
    const s=await api('/api/decode/start',{...decodeConfig,monitor_window_id:owner});
    if(revision!==directStartRevision){await api('/api/monitor/stop',{monitor_window_id:owner});await monitorRecorder.stop({complete:false,reason:'开始时已取消'});return;}
    lastCommand=0;update(s);
  }catch(error){
    await api('/api/monitor/stop',{monitor_window_id:owner}).catch(()=>{});
    await monitorRecorder.stop({complete:false,reason:'开始失败：'+error.message});
    throw error;
  }
}
$('start').addEventListener('click',()=>action($('start'),async()=>{
  const config=requestConfig();
  if(details?.requires_calibration&&!usesPreviousCalibration())throw new Error('请先完成上方校准，或选择使用上次完整校准。');
  config.calibration_audio=false;
  config.record_screen=workspaceKind==='six'&&$('recordScreen').checked;
  if(workspaceKind==='six'&&config.prompted.enabled)openSubjectWindow(config);
  else await startDirectDecoding(config);
}));
$('startCalibration').addEventListener('click',()=>action($('startCalibration'),async()=>{
  const audio=$('calibrationSound').checked;
  const revision=requestSubjectFocus('calibration');
  try{
    actionPrompts.cancel(state?.prompted?.session_id);
    await calibrationPrompts.prepare(audio,classes.map(taskName));
    if(revision!==subjectFocusRevision)return;
    const config={participant:$('participant').value,model_id:$('model').value,mode:$('mode').value,
      host:$('host').value,port:Number($('port').value),calibration_audio:audio,
      calibration_audio_ready:audio&&calibrationPrompts.prepared};
    const s=await api('/api/calibration/start',config);
    if(revision!==subjectFocusRevision){await api('/api/stop',{});return;}
    subjectFocusPending=null;lastCommand=0;update(s);
    showMessage('仅开始脑电校准；完成后保存并待命，再点击下方开始解码。');
  }finally{subjectFocusPending=null;renderSubjectFocus();}
}));
$('previewCalibrationAudio').addEventListener('click',()=>action($('previewCalibrationAudio'),async()=>{
  await calibrationPrompts.prepare(true,classes.map(taskName));
  const cue=await calibrationAudio.play(taskName(classes[0]));await cue.finished;
  text('brainAudioStatus','试听完成，请确认扬声器音量合适。');
}));
for(const id of ['stopCalibration','stopPrompted','stop','stopMonitor'])$(id).addEventListener('click',()=>stopDecoder().catch(e=>showMessage(e.message,true)));
$('fullscreenMonitor').addEventListener('click',requestMonitorFullscreen);
document.addEventListener('fullscreenchange',()=>{$('fullscreenMonitor').hidden=!!document.fullscreenElement;});
$('stopSubjectFocus').addEventListener('click',async()=>{
  subjectFocusRevision++;subjectFocusPending=null;
  try{await stopDecoder();}catch(e){showMessage(e.message,true);}finally{renderSubjectFocus();}
});
$('copyPath').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('savedPath').value);showMessage('保存路径已复制。');}catch{showMessage('浏览器未允许复制，请从路径框手动复制。',true);}});
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch(e){showMessage(e.message,true);}});
$('shutdown').addEventListener('click',async()=>{if(busy)return;calibrationPrompts.cancel(state?.calibration?.session_id);actionPrompts.cancel(state?.prompted?.session_id);try{await api('/api/shutdown',{});offline=true;updateControls();showMessage('本机服务正在关闭，运行中的会话会先停止并保存。');}catch(e){showMessage(e.message,true);}});
$('monitor').addEventListener('toggle',drawWave);window.addEventListener('resize',drawWave);
$('waveScale').addEventListener('change',drawWave);
window.addEventListener('pagehide',()=>{
  directStartRevision++;
  monitorRecorder.stop({complete:false,reason:'主窗口关闭或刷新'});
  if(directOwner)fetch('/api/monitor/stop',{method:'POST',keepalive:true,
    headers:{'Content-Type':'application/json','X-Online-Token':token},body:JSON.stringify({monitor_window_id:directOwner})}).catch(()=>{});
});
document.addEventListener('visibilitychange',()=>{
  if(document.visibilityState!=='visible'&&state?.running&&((state.prompted?.enabled&&!state.subject_window_id)||(state?.calibration?.required&&!state.calibration.ready))){
    stopDecoder().then(()=>showMessage('任务提示页面已离开前台，会话已停止并保留记录。')).catch(e=>showMessage(e.message,true));
  }
});
async function init(){try{models=await api('/api/models');const initial=await api('/api/state');workspaceKind=initial.running&&initial.mode==='four'?'four':'six';const preferred=models.find(m=>m.is_default);const preferredMode=preferred?({p01_six_acp_observation:'six',neuracle_acp:'acp',synthetic_tcp:'simulated',probability_demo:'demo'}[preferred.source]||'acp'):models.some(m=>m.source==='neuracle_acp')?'acp':'demo';$('mode').value=initial.running?initial.mode:preferredMode;lastSixMode=preferredMode;$('outputMode').value=initial.running?initial.output_mode||'simulation':'simulation';
  $('decisionMode').value=(initial.running&&initial.decision_settings?.mode)||(workspaceKind==='four'?'stable':'vote');
  $('cooldown').value=workspaceKind==='four'?'2':'4';
  if(initial.running&&initial.decision_settings){for(const [id,key] of Object.entries({voteWindow:'vote_window_seconds',voteRatio:'vote_ratio',decisionTimeout:'decision_timeout_seconds',cooldown:'cooldown_seconds',threshold:'confidence',margin:'margin',stable:'stable_seconds',release:'release_seconds'}))if(initial.decision_settings[key]!==undefined)$(id).value=initial.decision_settings[key];}
  if(initial.running&&initial.prompted?.enabled){const p=initial.prompted.settings||initial.prompted;$('collectionMode').value='prompted';for(const [id,key] of Object.entries({promptLoops:'loops',promptSeed:'seed',promptRest:'rest_seconds',promptPrepare:'prepare_seconds',promptMarking:'marking'}))if(p[key]!==undefined)$(id).value=p[key];if(p.trigger_port)$('promptPort').replaceChildren(option(p.trigger_port,p.trigger_port+' · 当前连接'));$('promptWireless').checked=!!p.trigger_host;if(p.trigger_host)$('promptDevice').replaceChildren(option(p.trigger_host,p.trigger_host+' · 当前连接'));}
  if(initial.running&&initial.output_config){const c=initial.output_config;if(c.port)$('handPort').replaceChildren(option(c.port,c.port+' · 当前连接'));for(const [id,key] of Object.entries({handPort:'port',handBaud:'baudrate',handId:'hand_id'}))if(c[key]!==undefined)$(id).value=c[key];}
  if(initial.running&&initial.eeg_endpoint){$('host').value=initial.eeg_endpoint.host;$('port').value=initial.eeg_endpoint.port;}
  if(!initial.running){$('host').value='';$('port').value=4097;}else if(initial.eeg_endpoint){const e=initial.eeg_endpoint;$('eegDevice').replaceChildren(option(e.host,e.host+' · 当前连接'));$('eegDevice').value=e.host;}
  if(!initial.running&&$('collectionMode').value==='prompted')syncPromptTiming('parts');
  workspaceFields();state=initial;buildDial(initial.classes);renderedSignature=JSON.stringify(initial.classes);update(initial);loadParticipants();poll();requestAnimationFrame(animate);if(!initial.running&&['six','four','acp'].includes($('mode').value)){refreshHandPorts().catch(e=>text('handCheckResult','串口搜索失败：'+e.message));searchEegDevices();}}catch(e){showMessage(e.message,true);}}
for(const id of ['decisionMode','voteWindow','voteRatio','decisionTimeout'])$(id).addEventListener('change',decisionFields);
$('cooldown').addEventListener('change',()=>{syncPromptTiming('cooldown');decisionFields();});
for(const id of ['promptRest','promptPrepare'])$(id).addEventListener('change',()=>syncPromptTiming('parts'));
$('collectionMode').addEventListener('change',()=>{if($('collectionMode').value==='prompted'){$('decisionMode').value='vote';syncPromptTiming('parts');}promptedFields();decisionFields();if(details?.requires_prompted_calibration)loadDetails();updateControls();searchTriggerDevices();});
$('promptMarking').addEventListener('change',()=>{promptedFields();updateControls();searchTriggerDevices();});
init();



$('importModel').addEventListener('click',async()=>{
  if(busy||state?.running)return;
  busy=true;$('importModel').disabled=true;
  try{
    const file=$('importFile').files[0];
    if(!file||file.size===0||file.size>32*1024*1024)throw Error('请选择不超过32MB的模型ZIP或PT文件。');
    const participant=$('importParticipant').value.trim();if(!participant)throw Error('请填写模型内部的被试编号。');
    text('importStatus','正在检查模型和输入约定…');
    const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=()=>reject(Error('读取模型文件失败'));reader.readAsDataURL(file);});
    const entry=await api('/api/import-model',{participant,name:$('importName').value.trim(),data});
    models=await api('/api/models');workspaceKind='six';lastSixMode=entry.source==='synthetic_tcp'?'simulated':'acp';$('mode').value=lastSixMode;
    workspaceFields();loadParticipants();$('participant').value=entry.participant_id;loadModels();$('model').value=entry.model_id;await loadDetails();
    text('importStatus','导入成功：'+entry.name+'；尚未连接设备。');
  }catch(e){text('importStatus','导入失败：'+e.message);showMessage(e.message,true);}
  finally{busy=false;$('importModel').disabled=false;updateControls();}
});

for(const id of ['host','port'])$(id).addEventListener('input',updateControls);

$('calibrationImportFolder').addEventListener('change',()=>{$('calibrationImportZip').value='';});
$('calibrationImportZip').addEventListener('change',()=>{$('calibrationImportFolder').value='';});
$('importCalibration').addEventListener('click',()=>action($('importCalibration'),async()=>{
  let upload=null;
  try{
    let selected=[];
    const zipped=$('calibrationImportZip').files[0];
    if(zipped)selected=[{name:'session.zip',file:zipped}];
    else{
      const files=[...$('calibrationImportFolder').files].filter(f=>['session.json','data.h5'].includes(f.name));
      const parents=new Set(files.map(f=>(f.webkitRelativePath||f.name).split('/').slice(0,-1).join('/')));
      if(files.length!==2||parents.size!==1||new Set(files.map(f=>f.name)).size!==2)throw Error('请选择单个采集会话文件夹，其中须包含session.json和data.h5。');
      selected=files.map(file=>({name:file.name,file}));
    }
    if(!selected.length)throw Error('请先选择采集会话文件夹或ZIP。');
    text('calibrationImportStatus','正在准备本机导入…');
    upload=await api('/api/calibration/import/open',{participant:$('participant').value,model_id:$('model').value,files:selected.map(x=>({name:x.name,size:x.file.size}))});
    const total=selected.reduce((n,x)=>n+x.file.size,0);let done=0;
    for(const {name,file} of selected){
      for(let offset=0;offset<file.size;offset+=upload.chunk_bytes){
        const chunk=file.slice(offset,offset+upload.chunk_bytes);
        const response=await fetch('/api/calibration/import/chunk?'+new URLSearchParams({id:upload.id,name,offset:String(offset)}),{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Online-Token':token},body:chunk});
        const result=await response.json();if(!response.ok)throw Error(result.error||'校准数据上传失败。');
        done+=chunk.size;text('calibrationImportStatus',`正在导入：${Math.round(done/total*100)}%`);
      }
    }
    text('calibrationImportStatus','正在检查任务标签、时长、通道与原始脑电，请稍候…');
    const result=await api('/api/calibration/import/finish',{id:upload.id});upload=null;
    try{localStorage.setItem('brain-calibration-choice:'+$('participant').value+':'+$('model').value,result.summary.id);}catch{}
    await loadDetails();
    if(details?.requires_calibration){$('savedCalibration').value=result.summary.id;$('brainCalibrationMode').value='reuse';calibrationChoiceChanged();}
    text('calibrationImportStatus',result.message);showMessage(result.message);
  }catch(error){text('calibrationImportStatus','导入失败：'+error.message);throw error;}
  finally{if(upload)await api('/api/calibration/import/cancel',{id:upload.id}).catch(()=>{});}
}));
