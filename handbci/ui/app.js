"use strict";
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="collection-token"]').content;
let profiles = [], selectedConfig = null, lastState = null, phaseSeen = null;
const maxClasses = 6;
let classTemplates = [], promptTemplates = [];
let receivedAt = performance.now(), polling = true, pollBusy = false, displayedReport = null;
let audioBusy = false, testingAudio = false, runGeneration = 0;
let triggerBusy = false;
let triggerDefaults = null;
const phases = {initial_rest:"开始前静息", prepare:"准备", task:"运动想象", rest:"休息", block_rest:"轮次间休息", finishing:"结束记录"};
const timingFields = {initialRest:"initial_rest_seconds", prepare:"prepare_seconds", taskDuration:"task_seconds", rest:"rest_seconds", blockRest:"block_rest_seconds"};

async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json","X-Collection-Token":token},body:JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}
function showError(message) {
  $("errorBox").textContent=message; $("errorBox").hidden=!message;
  if (message && subjectView.visible) subjectView.leave().catch(()=>{});
}
const cueAudio = new CollectionAudio(api, message => { $("audioStatus").textContent = message; });
const subjectView = new StaticSubjectView($("subjectView"), $("subjectText"), document, $("subjectRounds"));
const defaultPromptNames = {
  "想象左手反复伸腕":"左手", "想象右手反复伸腕":"右手",
  "想象双手同时反复伸腕":"双手", "想象双脚同时反复抬脚尖":"双脚",
  "想象左手伸腕与右脚抬脚尖同时进行":"左手与右脚",
  "想象右手伸腕与左脚抬脚尖同时进行":"右手与左脚"
};
function updateControls() {
  const running = Boolean(lastState?.running);
  const busy = running || audioBusy || triggerBusy || Boolean(lastState?.trigger_probe_busy);
  $("start").disabled = busy;
  for (const element of $("settings").querySelectorAll("input,select,button")) element.disabled = busy;
  const simulated = selectedConfig?.device.source === "synthetic_tcp";
  const network = $("markingMode").value === "network";
  const serial = $("markingMode").value === "hardware";
  $("markingMode").disabled = busy || simulated;
  $("triggerPort").disabled = busy || simulated || !serial;
  $("triggerHost").disabled = busy || simulated || !network;
  $("triggerTcpPort").disabled = busy || simulated || !network;
  $("testTriggerBox").disabled = busy || simulated || (!serial && !network);
  $("refreshPorts").disabled = busy || simulated || !serial;
  $("readTriggerNetwork").disabled = busy || simulated || !network;
  $("serialPortField").hidden = !serial;
  $("refreshPorts").hidden = !serial;
  $("networkHostField").hidden = !network;
  $("networkPortField").hidden = !network;
  $("readTriggerNetwork").hidden = !network;
  $("addClass").disabled = busy || !selectedConfig || selectedConfig.classes.length >= maxClasses;
  for (const element of $("taskList").querySelectorAll(".remove-class"))
    element.disabled = busy || selectedConfig.classes.length <= 2;
  $("audioEnabled").disabled = busy;
  $("audioVolume").disabled = busy;
  $("audioTest").disabled = running || triggerBusy || Boolean(lastState?.trigger_probe_busy) || (audioBusy && !testingAudio);
  $("audioTest").textContent = testingAudio ? "停止试听" : "试听提示";
}
async function phaseAudioError(p, generation, error) {
  if (generation !== runGeneration || error.name === "AbortError" || !lastState?.running) return;
  showError("声音提示失败，采集将停止：" + error.message);
  cueAudio.cancel();
  try { await api("/api/audio-error", {phase_id:p.id, error:error.message}); }
  catch (_) { await api("/api/stop", {}).catch(e => showError(e.message)); }
}
async function acknowledgePhase(p, generation) {
  if (generation !== runGeneration || !lastState?.running || lastState.phase?.id !== p.id) return;
  const renderTime = performance.timeOrigin + performance.now();
  try {
    let audio = {enabled:false};
    if (lastState?.audio?.enabled) {
      const text = p.phase === "task" ? "滴" : p.phase === "prepare" ? (p.prompt_text || p.task_text) :
        p.phase === "initial_rest" ? "开始" : p.phase === "finishing" ? "结束" : "休息";
      const clip = await cueAudio.play(text, p.phase === "task" ? "beep" : "speech");
      if (p.phase === "prepare" || p.phase === "finishing") await clip.finished;
      else clip.finished.catch(error => phaseAudioError(p, generation, error));
      audio = {...clip.metadata};
    }
    if (generation !== runGeneration || !lastState?.running || lastState.phase?.id !== p.id) return;
    await api("/api/ack", {phase_id:p.id, render_time_ms:renderTime, audio,
      prepare_elapsed_seconds:p.phase==="prepare"?(performance.timeOrigin+performance.now()-renderTime)/1000:0});
  } catch (error) {
    if (lastState?.audio?.enabled) await phaseAudioError(p, generation, error);
    else showError(error.message);
  }
}
function selectConfig(config) {
  if (!Array.isArray(config.classes) || config.classes.length < 2 || config.classes.length > maxClasses)
    throw new Error("采集界面支持2至6类任务。");
  selectedConfig = structuredClone(config);
  classTemplates = structuredClone(config.classes);
  promptTemplates = config.classes.map((c,i)=>config.session_metadata?.prompt_texts?.[i] || defaultPromptNames[c.task_text] || c.task_text);
  selectedConfig.session_metadata = {...selectedConfig.session_metadata, prompt_texts:[...promptTemplates]};
  const c=config.collection,d=config.device;
  $("markingMode").value=d.source==="synthetic_tcp"?"software":d.trigger_connection?.mode==="network"?"network":d.trigger_port?"hardware":"software";
  $("triggerPort").value=d.trigger_port||"";
  $("triggerHost").value=d.trigger_connection?.host||triggerDefaults?.host||"";
  $("triggerTcpPort").value=d.trigger_connection?.port||4321;
  $("triggerStatus").textContent=d.source==="synthetic_tcp"?"模拟源使用模拟触发，不连接真实TriggerBox。":"按实际连接选择Wi-Fi/TCP或USB串口；当前配置尚未测试TriggerBox。";
  $("loops").value=c.loops;
  for (const [id,key] of Object.entries(timingFields)) $(id).value=c[key] ?? 0;
  $("signalInfo").textContent=`${d.channels.length} 路 · ${d.sample_rate} Hz`;
  $("deviceDetails").textContent=`${d.host || "127.0.0.1"}:${d.port || 8712} · 流通道 ${d.stream_channels}`;
  renderTasks();
  const simulated=d.source === "synthetic_tcp";
  $("sourceBadge").textContent=simulated?"模拟数据":"真实设备配置";
  updateMarkingNote();
  estimate();
}
function updateMarkingNote() {
  const simulated=selectedConfig?.device.source==="synthetic_tcp";
  const mode=$("markingMode").value;
  const marking=mode==="network"?"TriggerBox Wi-Fi/TCP，须先确认DCP兼容":mode==="hardware"?"TriggerBox USB串口":"软件接收时钟（近似）";
  $("modeNote").textContent=simulated?"人工信号通过本机 TCP 端口采集，无需连接外设。":`先在厂家软件开启脑电转发。标记：${marking}。`;
}
function readTriggerSettings() {
  const mode=$("markingMode").value,port=$("triggerPort").value.trim().toUpperCase();
  if(mode==="network") {
    const host=$("triggerHost").value.trim(),tcpPort=Number($("triggerTcpPort").value);
    if(!/^\d{1,3}(\.\d{1,3}){3}$/.test(host)||host.split(".").some(x=>Number(x)>255))throw new Error("请填写TriggerBox实际局域网IPv4地址。");
    if(!Number.isInteger(tcpPort)||tcpPort<1||tcpPort>65535)throw new Error("DCP打标端口须为1至65535的整数。");
    if([4097,8712].includes(tcpPort))throw new Error("4097是设备管理端口，8712是脑电转发端口；不能当DCP打标端口。手册示例是4321。");
    return {mode,host,port:tcpPort};
  }
  if(mode==="hardware" && !/^COM[1-9][0-9]{0,3}$/.test(port))throw new Error("请填写TriggerBox实际COM串口，例如COM3；不是8712。");
  return {mode,port:mode==="hardware"?port:""};
}
async function readTriggerNetworkDefaults() {
  triggerBusy=true;updateControls();
  try {
    triggerDefaults=await api("/api/triggerbox/defaults");
    $("triggerHost").value=triggerDefaults.host||"";
    $("triggerTcpPort").value=triggerDefaults.port;
    if($("markingMode").value==="network")$("triggerStatus").textContent=triggerDefaults.message;
  } catch(e){$("triggerStatus").textContent="无法读取厂家保存IP，可手动填写："+e.message;}
  finally {triggerBusy=false;updateControls();}
}
function triggerEndpoint(info) { return info?.endpoint || (info?.host?`${info.host}:${info.port}`:info?.port||""); }
async function refreshTriggerPorts() {
  triggerBusy=true;updateControls();
  try {
    const result=await api("/api/triggerbox/ports");
    $("triggerPorts").replaceChildren();
    for(const p of result.ports){const option=document.createElement("option");option.value=p.port;option.label=p.description;$("triggerPorts").append(option);}
    $("triggerStatus").textContent=result.message;
  } catch(e){$("triggerStatus").textContent="无法列出串口："+e.message;}
  finally {triggerBusy=false;updateControls();}
}
function readTasks() {
  const texts=[...$("taskList").querySelectorAll(".task-input")].map(input=>input.value.trim());
  const prompts=[...$("taskList").querySelectorAll(".prompt-input")].map(input=>input.value.trim());
  selectedConfig.classes.forEach((c,i)=>{c.task_text=texts[i];c.id=i;});
  selectedConfig.session_metadata.prompt_texts=prompts;
  return {texts,prompts};
}
function renderTasks() {
  $("classCount").value=selectedConfig.classes.length;
  $("taskList").replaceChildren();
  selectedConfig.classes.forEach((c,i)=>{
    c.id=i;
    const label=document.createElement("label");label.textContent=`类别 ${i+1}`;
    const input=document.createElement("input");input.value=c.task_text;input.maxLength=100;input.className="task-input";
    label.textContent="类别 "+(i+1)+" · 任务说明";
    label.append(input);
    const entry=document.createElement("div");entry.className="task-entry";
    const heading=document.createElement("div");heading.className="task-heading";
    const title=document.createElement("strong");title.textContent=`类别 ${i+1} · 标签 ${i}`;
    const remove=document.createElement("button");remove.type="button";remove.textContent="删除";remove.className="quiet remove-class";
    remove.setAttribute("aria-label",`删除类别 ${i+1}`);
    remove.addEventListener("click",()=>{
      if(lastState?.running || audioBusy || selectedConfig.classes.length<=2)return;
      readTasks();selectedConfig.classes.splice(i,1);selectedConfig.session_metadata.prompt_texts.splice(i,1);
      renderTasks();estimate();showError("");
    });
    heading.append(title,remove);
    const promptLabel=document.createElement("label");promptLabel.textContent="类别 "+(i+1)+" · 屏幕/语音名称";
    const promptInput=document.createElement("input");
    promptInput.value=selectedConfig.session_metadata.prompt_texts[i];
    promptInput.maxLength=100;promptInput.className="prompt-input";
    promptLabel.append(promptInput);entry.append(heading,promptLabel,label);$("taskList").append(entry);
  });
  updateControls();
}
function resizeClasses(count) {
  if(lastState?.running || audioBusy)return;
  if(!Number.isInteger(count) || count<2 || count>maxClasses) {
    $("classCount").value=selectedConfig.classes.length;
    showError("类别数量须为2至6的整数。");return;
  }
  readTasks();
  selectedConfig.classes.splice(count);
  selectedConfig.session_metadata.prompt_texts.splice(count);
  while(selectedConfig.classes.length<count) {
    const i=selectedConfig.classes.length;
    const template=classTemplates.findIndex(c=>!selectedConfig.classes.some(existing=>existing.name===c.name));
    let c,prompt;
    if(template>=0) {c=structuredClone(classTemplates[template]);prompt=promptTemplates[template];}
    else {
      let serial=i+1;
      while(selectedConfig.classes.some(c=>c.name===`custom_class_${serial}`))serial++;
      c={id:i,name:`custom_class_${serial}`,task_text:`想象任务 ${serial}`,action:["thumb","index","middle","ring","little","pinch"][i]};
      prompt=`任务 ${serial}`;
    }
    c.id=i;selectedConfig.classes.push(c);selectedConfig.session_metadata.prompt_texts.push(prompt);
  }
  renderTasks();showError("");
  estimate();
}
function estimate() {
  if(!selectedConfig)return;
  const loops=Number($("loops").value),n=selectedConfig.classes.length;
  const duration=Number($("initialRest").value)+loops*n*(Number($("prepare").value)+Number($("taskDuration").value)+Number($("rest").value))+Math.max(0,loops-1)*Number($("blockRest").value)+(selectedConfig.collection.tail_seconds??1);
  $("estimate").textContent=`共 ${loops*n} 个试次，每类 ${loops} 次；预计约 ${Math.ceil(duration)} 秒，显示和触发确认可能增加少量时间。`;
  if(!lastState?.running)$("trialCounter").textContent=`${n} 类任务 · 每类 ${loops} 次`;
}
function render(state) {
  const wasRunning = lastState?.running;
  lastState=state;receivedAt=performance.now();
  const running=state.running;
  if (running || !audioBusy) subjectView.update(state);
  if (["failed","cancelled"].includes(state.status) && subjectView.visible) subjectView.leave().catch(()=>{});
  if (running || (audioBusy && !testingAudio)) $("result").hidden=true;
  if (wasRunning && !running) {
    runGeneration++; cueAudio.cancel();
    $("audioStatus").textContent = state.audio?.enabled ? "本次声音提示已结束" : "声音提示已关闭";
  }
  const titles={idle:"等待开始",connecting:"正在连接并开始记录…",collecting:"采集中",stopping:"正在停止并保存…",completed:"采集完成",cancelled:"已停止 · 保留未完成记录",failed:"采集未完成"};
  $("statusText").textContent=titles[state.status]||state.status;
  updateControls();$("stop").disabled=!running||state.status==="stopping";
  $("sampleCount").textContent=Number(state.samples_received||0).toLocaleString();
  $("recordSeconds").textContent=`${((state.samples_received||0)/(state.sample_rate||selectedConfig?.device.sample_rate||1000)).toFixed(1)} 秒`;
  if(running && state.triggerbox?.enabled)$("triggerStatus").textContent=`TriggerBox ${triggerEndpoint(state.triggerbox)} · ${state.triggerbox.device_name||"正在连接"}；已在脑电流中确认 ${state.triggerbox.verified_events} 条事件。`;
  if(state.error)showError(state.error);
  const p=state.phase;
  if(running&&p) {
    $("phaseName").textContent=phases[p.phase]||p.phase;
    $("trialCounter").textContent=p.trial_id?`第 ${p.loop||p.loops} 轮 · 试次 ${p.trial_id} / ${p.total_trials}`:"尚未进入想象任务";
    const title=p.phase==="task"?p.task_text:p.phase==="prepare"?"准备下一项任务":phases[p.phase];
    $("cueTitle").textContent=title;
    $("cueSubtitle").textContent=p.phase==="task"?"只想象运动的感觉，保持身体放松，不实际运动。":p.phase==="prepare"?`下一项：${p.task_text}。请先放松，等待“运动想象”阶段再开始。`:p.phase==="finishing"?"本轮任务结束，正在完成记录，请稍候。":"停止想象，自然呼吸，保持放松。";
    if(p.awaiting_ack&&phaseSeen!==p.id&&document.visibilityState==="visible") {
      phaseSeen=p.id;
      const generation = runGeneration;
      requestAnimationFrame(()=>requestAnimationFrame(()=>acknowledgePhase(p, generation)));
    }
  } else if(running) {
    $("cueTitle").textContent="正在连接数据源";$("cueSubtitle").textContent="收到连续信号后开始静息和任务提示。";
    $("phaseName").textContent="连接中";$("countdown").textContent="—";$("timerLabel").textContent="请稍候";
  } else if(state.status!=="idle") {
    $("phaseName").textContent="本次采集结束";$("cueTitle").textContent=state.status==="completed"?"采集完成":"采集已停止";
    $("cueSubtitle").textContent=state.status==="completed"?"可以放松了。各类完成次数与保存位置见下方。":"已收到的数据会保留；本次会话标记为未完成。";
    $("countdown").textContent="—";$("timerLabel").textContent="本次记录已结束";
    $("phaseProgress").value=state.status==="completed"?1:0;
  }
  if(!running&&!audioBusy&&(state.report||state.session)&&displayedReport!==state.session+state.status) {
    displayedReport=state.session+state.status;$("result").hidden=false;
    const r=state.report;
    $("audioResult").textContent=r?.audio?.enabled?"声音提示已记录：任务播报 "+r.audio.prepared_task_speeches+" 次，开始滴声 "+r.audio.task_beeps+" 次。":"本次未开启声音提示。";
    $("triggerResult").textContent=r?.triggerbox?.enabled?`TriggerBox ${triggerEndpoint(r.triggerbox)} · ${r.triggerbox.device_name||"设备名未读取"}；已在脑电流中确认 ${r.triggerbox.verified_events} 条事件。${r.complete?"":"本次记录未完成，请查看错误。"}`:r?.source==="synthetic_tcp"?"本次使用模拟触发。":"本次使用软件近似标记，未连接TriggerBox。";
    $("resultTitle").textContent=r?.complete?"已保存完整采集记录":"已保留未完成记录";
    $("resultSummary").textContent=r?`${r.eeg_channels} 路信号，${r.signal_duration_seconds.toFixed(2)} 秒；${r.n_trials} 个完整试次，${r.n_events} 条事件。${r.source==="synthetic_tcp"?"所有信号均为人工模拟。":""}`:"连接或开始阶段中止；请检查文件状态，不直接用于训练。";
    $("resultPath").value=state.session||"";$("resultRows").replaceChildren();
    (r?.tasks||[]).forEach(task=>{const row=document.createElement("tr");for(const value of [task.task_text,task.completed_trials]){const cell=document.createElement("td");cell.textContent=value;row.append(cell);}$("resultRows").append(row);});
  }
}
function timer() {
  const p=lastState?.phase;
  if(lastState?.running&&p&&!subjectView.visible) {
    const remaining=Math.max(0,p.remaining_seconds-(p.awaiting_ack?0:(performance.now()-receivedAt)/1000));
    $("countdown").textContent=Math.ceil(remaining);
    $("timerLabel").textContent=p.awaiting_ack?(lastState.audio?.enabled&&p.phase==="prepare"?"正在播报任务，请保持放松":"正在同步提示与记录"):"秒后进入下一阶段";
    $("phaseProgress").value=p.duration_seconds?Math.min(1,1-remaining/p.duration_seconds):1;
  }
  requestAnimationFrame(timer);
}
async function poll() {
  if(!polling||pollBusy)return;
  pollBusy=true;
  try {render(await api("/api/state"));}catch(e){showError(`采集界面与本机程序连接失败：${e.message}。请检查启动窗口。`);}finally{pollBusy=false;}
}
$("profile").addEventListener("change",()=>{const profile=profiles.find(p=>p.id===$("profile").value);if(profile)selectConfig(profile.config);});
$("markingMode").addEventListener("change",()=>{updateMarkingNote();updateControls();const mode=$("markingMode").value;$("triggerStatus").textContent=mode==="network"?(triggerDefaults?.message||"填写设备IP及DCP打标端口；本型号兼容性尚未验证。请先测试连接。"):mode==="hardware"?"请选择实际COM串口并测试连接；尚未验证脑电触发链路。":"本次将使用软件近似标记。";});
$("triggerPort").addEventListener("input",()=>{$("triggerStatus").textContent="串口已更改，需要重新测试连接。";});
for(const id of ["triggerHost","triggerTcpPort"])$(id).addEventListener("input",()=>{$("triggerStatus").textContent="网络参数已更改，需要重新测试DCP连接。";});
$("refreshPorts").addEventListener("click",refreshTriggerPorts);
$("readTriggerNetwork").addEventListener("click",readTriggerNetworkDefaults);
$("testTriggerBox").addEventListener("click",async()=>{
  showError("");triggerBusy=true;updateControls();
  try {const settings=readTriggerSettings();$("triggerStatus").textContent="正在读取TriggerBox设备名称…";const result=await api("/api/triggerbox/test",settings.mode==="network"?{connection:settings}:{port:settings.port});$("triggerStatus").textContent=result.message;}
  catch(e){$("triggerStatus").textContent=e.message;showError(e.message);}
  finally {triggerBusy=false;updateControls();}
});
$("classCount").addEventListener("change",()=>resizeClasses(Number($("classCount").value)));
$("addClass").addEventListener("click",()=>resizeClasses(selectedConfig.classes.length+1));
for(const id of ["loops",...Object.keys(timingFields)])$(id).addEventListener("input",estimate);
$("start").addEventListener("click",async()=>{
  showError("");$("result").hidden=true;displayedReport=null;phaseSeen=null;
  audioBusy=true;updateControls();cueAudio.cancel();runGeneration++;
  const collection={};for(const [id,key] of Object.entries(timingFields))collection[key]=Number($(id).value);
  const texts=[...document.querySelectorAll(".task-input")].map(i=>i.value.trim());
  const prompts=[...document.querySelectorAll(".prompt-input")].map(i=>i.value.trim());
  const audio={enabled:$("audioEnabled").checked,volume:Number($("audioVolume").value)};
  try {
    const triggerbox=readTriggerSettings();
    if(!Number.isInteger(Number($("classCount").value)) || Number($("classCount").value)!==selectedConfig.classes.length)
      throw new Error("请先设置2至6的整数类别数量。");
    readTasks();
    if([...texts,...prompts].some(text=>!text || text.length>100))throw new Error("每类任务名称和说明须为1至100个字符。");
    if(new Set(prompts).size!==prompts.length)throw new Error("不同类别的屏幕/语音名称不能重复；相同任务请合并为一类并增加采集轮数。");
    subjectView.update({running:false,status:"idle"});
    // Both calls must happen in the click's user-activation scope.
    const soundReady=audio.enabled?cueAudio.unlock():Promise.resolve();
    const fullscreenReady=subjectView.enter();
    await Promise.all([soundReady,fullscreenReady]);
    if(audio.enabled) {
      cueAudio.volume=audio.volume/100;
      await cueAudio.prepare(prompts);
    } else $("audioStatus").textContent="声音提示已关闭，仅使用屏幕提示";
    const state=await api("/api/start",{config:selectedConfig,participant:$("participant").value,loops:Number($("loops").value),collection,task_texts:texts,prompt_texts:prompts,audio,triggerbox});
    $("settings").open=false;render(state);$("cueCard").scrollIntoView({block:"start"});
  } catch(e){cueAudio.cancel();await subjectView.leave().catch(()=>{});showError(e.message);}
  finally {audioBusy=false;updateControls();}
});
$("stop").addEventListener("click",()=>{runGeneration++;cueAudio.cancel();api("/api/stop",{}).catch(e=>showError(e.message));});
$("audioVolume").addEventListener("input",()=>{$("audioVolumeValue").textContent=$("audioVolume").value+"%";});
$("audioEnabled").addEventListener("change",()=>{$("audioStatus").textContent=$("audioEnabled").checked?"声音已开启；开始前请试听":"声音已关闭，仅使用屏幕提示";});
$("audioTest").addEventListener("click",async()=>{
  if(testingAudio){cueAudio.cancel();return;}
  showError("");audioBusy=true;testingAudio=true;updateControls();cueAudio.cancel();
  try {
    cueAudio.volume=Number($("audioVolume").value)/100;
    await cueAudio.unlock();
    const text=document.querySelector(".prompt-input")?.value.trim()||"左手";
    await cueAudio.prepare([text]);
    for(const [word,kind] of [["开始","speech"],[text,"speech"],["滴","beep"],["休息","speech"],["结束","speech"]]) {
      const clip=await cueAudio.play(word,kind);await clip.finished;
    }
    $("audioStatus").textContent="试听播放已完成，请确认你能听到语音与滴声";
  } catch(e){if(e.name!=="AbortError"){showError(e.message);$("audioStatus").textContent="声音未就绪，请检查扬声器和本机中文语音";}else $("audioStatus").textContent="已停止试听";}
  finally {audioBusy=false;testingAudio=false;updateControls();}
});
$("fullscreen").addEventListener("click",async()=>{
  try {if(subjectView.visible)await subjectView.leave();else{subjectView.update(lastState||{});await subjectView.enter();}}
  catch(e){showError(e.message);}
});
$("shutdown").addEventListener("click",async()=>{try{await api("/api/shutdown",{});polling=false;$("statusText").textContent="正在关闭；采集文件会先完成保存";$("start").disabled=true;$("stop").disabled=true;$("cueTitle").textContent="程序正在关闭";$("cueSubtitle").textContent="保存结束后启动窗口会退出，可关闭本页面。";}catch(e){showError(e.message);}});
$("copyPath").addEventListener("click",async()=>{try{await navigator.clipboard.writeText($("resultPath").value);$("copyPath").textContent="已复制";}catch(e){$("resultPath").select();showError("请复制已选中的保存位置。");}});
$("importButton").addEventListener("click",()=>$("importFile").click());
$("importFile").addEventListener("change",async()=>{try{const file=$("importFile").files[0];if(!file)return;if(file.size>100000)throw new Error("配置文件过大。");const config=JSON.parse((await file.text()).replace(/^\uFEFF/,""));if(!config.device||!config.collection||!Array.isArray(config.classes))throw new Error("配置缺少设备、采集或任务字段。");selectConfig(config);$("profile").value="";showError("");}catch(e){showError(`无法导入配置：${e.message}`);}});
async function init(){try{profiles=await api("/api/profiles");for(const p of profiles){const option=document.createElement("option");option.value=p.id;option.textContent=p.title;$("profile").append(option);}const initial=profiles.find(p=>p.id==="neuracle32")||profiles[0];$("profile").value=initial.id;selectConfig(initial.config);await readTriggerNetworkDefaults();await poll();setInterval(poll,100);timer();}catch(e){showError(`无法初始化界面：${e.message}`);$("start").disabled=true;}}
init();
