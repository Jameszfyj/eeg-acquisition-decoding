"use strict";
const $=id=>document.getElementById(id);
const token=document.querySelector('meta[name="online-token"]').content;
let models=[], details=null, state=null, polling=true, busy=false, previousCommands=-1, resultKey="";
const phaseNames={idle:"等待开始",warming:"正在准备滤波与滑动窗口",decoding:"有效判断时间",interval:"判断间隔",finished:"本次运行结束"};
async function api(path,payload){
  const response=await fetch(path,payload===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json","X-Online-Token":token},body:JSON.stringify(payload)});
  const result=await response.json();if(!response.ok)throw new Error(result.error||"请求失败");return result;
}
function error(message){$("errorBox").textContent=message;$("errorBox").hidden=!message;}
function note(message){$("notice").textContent=message;$("notice").hidden=!message;}
function addOption(select,value,text){const option=document.createElement("option");option.value=value;option.textContent=text;select.append(option);}
async function loadModels(preferredParticipant,preferredModel){
  models=await api("/api/models");const old=preferredParticipant||$("participant").value;
  $("participant").replaceChildren();[...new Set(models.map(m=>m.participant_id))].sort().forEach(p=>addOption($("participant"),p,p));
  if([...$("participant").options].some(o=>o.value===old))$("participant").value=old;
  if(!models.length){details=null;$("start").disabled=true;$("check").disabled=true;note("尚无已登记模型。请在页面下方导入对应被试的整理版模型。");return;}
  await loadParticipant(preferredModel);
}
async function loadParticipant(preferredModel){
  const available=models.filter(m=>m.participant_id===$("participant").value);$("model").replaceChildren();
  available.forEach(m=>addOption($("model"),m.model_id,m.name||m.model_id));
  if(available.some(m=>m.model_id===preferredModel))$("model").value=preferredModel;
  await loadDetails();
}
async function loadDetails(){
  details=null;$("start").disabled=true;
  if(!$("model").value)return;
  try{
    details=await api("/api/model?participant="+encodeURIComponent($("participant").value)+"&model_id="+encodeURIComponent($("model").value));
    $("mode").options[0].disabled=!details.synthetic_model;$("mode").options[1].disabled=details.synthetic_model;
    $("mode").value=details.synthetic_model?"simulated":"device";
    $("host").value=details.host||"127.0.0.1";$("port").value=details.port||8712;$("modelPath").value=details.model;
    $("modelInfo").textContent=details.n_channels+" 路 · "+details.sample_rate+" Hz · "+details.n_classes+" 类 · "+details.window_seconds+" 秒滑动窗口";
    $("decisionInfo").textContent="输出条件：最高概率 ≥ "+Math.round(details.decision.confidence*100)+"%，领先第二类 ≥ "+Math.round(details.decision.margin*100)+"%，持续稳定 "+details.decision.stable_seconds+" 秒。参数随所选模型加载。";
    $("probabilityTitle").textContent=details.n_classes+" 类预测概率";
    $("simulationTask").replaceChildren();$("probabilityRows").replaceChildren();
    details.classes.forEach(c=>{
      addOption($("simulationTask"),String(c.id),c.task_text);
      const row=document.createElement("div");row.className="probability-row";row.dataset.label=c.id;
      const label=document.createElement("div");label.className="probability-label";
      const text=document.createElement("span");text.textContent=c.task_text;
      const value=document.createElement("strong");value.textContent="—";label.append(text,value);
      const bar=document.createElement("progress");bar.max=1;bar.value=0;bar.setAttribute("aria-label",c.task_text+"预测概率");
      row.append(label,bar);$("probabilityRows").append(row);
    });
    if(details.classes.length>1)$("simulationTask").value="1";
    $("start").disabled=!!state?.running;$("check").disabled=!!state?.running;
    previousCommands=-1;if(!state?.running){$("result").hidden=true;resultKey="";}
    error("");updateMode();
  }catch(e){error(e.message);}
}
function updateMode(){
  const simulated=$("mode").value==="simulated";$("simulationControl").hidden=!simulated;
  $("host").disabled=simulated||!!state?.running;$("port").disabled=simulated||!!state?.running;
  $("sourceBadge").textContent=simulated?"人工信号演示":"真实设备";
  $("modeNote").textContent=simulated?"自动启动本机 TCP 人工信号源，加载所选模型持续预测。":"先在厂家软件中启动数据转发，再开始在线解码。";
  note(simulated?"当前使用人工模型与人工信号。切换“模拟信号任务”会改变发生器输入；识别结果由模型根据收到的信号计算。":"使用所选被试的真实脑电模型；通道、单位、预处理和任务定义必须与训练时一致。");
}
function requestConfig(){return {participant:$("participant").value,model_id:$("model").value,mode:$("mode").value,trials:Number($("trials").value),host:$("host").value,port:Number($("port").value),simulation_label:Number($("simulationTask").value)};}
function render(next){
  if(!next.running&&next.selection&&next.selection.model_id!==$("model").value){
    next={status:"idle",running:false,samples_received:0,n_predictions:0,commands:[],probabilities:[],phase:"idle",trial:0,trace:[],report:null};
  }
  state=next;const running=next.running;
  const titles={idle:"等待开始",connecting:"正在连接数据源",running:"在线解码中",stopping:"正在停止并保存",completed:"本次在线解码完成",stopped:"已停止并保留记录",failed:"运行未完成"};
  $("statusText").textContent=titles[next.status]||next.status;
  $("start").disabled=running||!details;$("check").disabled=running||!details;$("stop").disabled=!running||next.status==="stopping";
  for(const id of ["participant","model","trials","mode","importParticipant","importName","modelFile","importModel"])$(id).disabled=running;
  $("simulationTask").disabled=!details||$("mode").value!=="simulated";
  $("host").disabled=running||$("mode").value==="simulated";$("port").disabled=running||$("mode").value==="simulated";
  $("samples").textContent=Number(next.samples_received||0).toLocaleString();
  $("duration").textContent=((next.samples_received||0)/(next.sample_rate||details?.sample_rate||1000)).toFixed(1)+" 秒";
  $("predictions").textContent=Number(next.n_predictions||0).toLocaleString();
  $("inference").textContent=next.n_predictions?Number(next.inference_ms).toFixed(1)+" ms":"—";
  $("phaseText").textContent=phaseNames[next.phase]||"等待开始";
  $("trialText").textContent=next.trial?"第 "+next.trial+" / "+next.total_trials+" 次判断":"等待开始";
  const probabilities=next.probabilities||[];
  if(probabilities.length&&details){
    let top=0;probabilities.forEach((p,i)=>{if(p>probabilities[top])top=i;});
    $("predictionTask").textContent=(next.classes||details.classes)[top].task_text;
    $("confidence").textContent=(probabilities[top]*100).toFixed(1)+"%";
    [...$("probabilityRows").children].forEach((row,i)=>{row.classList.toggle("top",i===top);row.querySelector("progress").value=probabilities[i]||0;row.querySelector("strong").textContent=((probabilities[i]||0)*100).toFixed(1)+"%";});
    $("updateHint").textContent=running?"持续更新":"本次最后结果";
    $("decisionStatus").textContent=next.phase==="decoding"?"当前判断剩余约 "+Number(next.trial_remaining||0).toFixed(1)+" 秒；等待满足稳定输出条件。":next.phase==="warming"?"正在积累窗口，稍后进入有效判断时间。":next.phase==="interval"?"本次判断已结束，等待下一次；期间仍持续接收和预测。":"本次记录已结束，可查看下方保存位置。";
  }else if(running){$("predictionTask").textContent="正在积累信号";$("confidence").textContent="—";$("decisionStatus").textContent="滤波启动和完整窗口准备好后开始预测。";}
  else{$("predictionTask").textContent="等待实时信号";$("confidence").textContent="—";$("updateHint").textContent="等待更新";$("decisionStatus").textContent="预处理和预测会随新数据持续更新。";[...$("probabilityRows").children].forEach(row=>{row.classList.remove("top");row.querySelector("progress").value=0;row.querySelector("strong").textContent="—";});}
  if(next.channel_name)$("waveTitle").textContent="原始信号预览 · "+next.channel_name+" · "+next.unit;
  const commands=next.commands||[];
  $("commandCount").textContent=commands.length+" 条指令";
  $("lastCommand").textContent=commands.length?commands[commands.length-1].action_text:"尚未输出";
  if(commands.length!==previousCommands){previousCommands=commands.length;$("commandRows").replaceChildren();
    if(!commands.length){const row=document.createElement("tr"),cell=document.createElement("td");cell.colSpan=5;cell.className="empty";cell.textContent="达到输出条件后，指令会显示在这里。";row.append(cell);$("commandRows").append(row);}
    commands.slice(-20).reverse().forEach(c=>{const row=document.createElement("tr");[Number(c.decision_time).toFixed(2)+" 秒",c.trial_id,c.task_text,c.action_text,(c.confidence*100).toFixed(1)+"%"].forEach(value=>{const cell=document.createElement("td");cell.textContent=value;row.append(cell);});$("commandRows").append(row);});
  }
  if(next.status==="failed")error(next.error);
  if(!running&&next.report&&resultKey!==next.session+next.status){resultKey=next.session+next.status;$("result").hidden=false;
    $("resultTitle").textContent=next.report.complete?"在线记录已保存":"已保留未完成的在线记录";
    $("resultSummary").textContent=next.report.duration_seconds.toFixed(2)+" 秒；"+next.report.n_predictions+" 次预测，"+next.report.n_commands+" 条模拟指令。"+(next.mode==="simulated"?"本次信号全部为人工模拟。":"")+"在线记录没有已知任务标签，不会自动进入训练。";
    $("resultPath").value=next.session||"";
  }
  drawWave(next.trace||[]);
}
function drawWave(points){
  const canvas=$("waveform"),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
  const width=Math.max(1,Math.floor(rect.width*ratio)),height=Math.max(1,Math.floor(rect.height*ratio));
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
  const ctx=canvas.getContext("2d");ctx.setTransform(ratio,0,0,ratio,0,0);const w=rect.width,h=rect.height;
  ctx.clearRect(0,0,w,h);ctx.strokeStyle="#edf2f6";ctx.lineWidth=1;
  for(let y=1;y<4;y++){ctx.beginPath();ctx.moveTo(14,h*y/4);ctx.lineTo(w-14,h*y/4);ctx.stroke();}
  if(points.length<2){ctx.fillStyle="#93a4b1";ctx.font='12px "Microsoft YaHei", sans-serif';ctx.fillText("接收数据后显示实时波形",22,h/2);return;}
  const minTime=points[0][0],maxTime=points[points.length-1][0],ys=points.map(p=>p[1]);
  let low=Math.min(...ys),high=Math.max(...ys);const range=Math.max(1e-6,high-low);low-=range*.1;high+=range*.1;
  ctx.strokeStyle="#15909c";ctx.lineWidth=1.4;ctx.beginPath();points.forEach((p,i)=>{const x=15+(p[0]-minTime)/Math.max(.01,maxTime-minTime)*(w-30),y=h-14-(p[1]-low)/(high-low)*(h-28);if(i&&p[0]-points[i-1][0]<=Math.max(1,Math.floor((state?.sample_rate||1000)/80))/(state?.sample_rate||1000)+.000501)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();
}
async function poll(){if(!polling||busy)return;busy=true;try{render(await api("/api/state"));}catch(e){error("页面与本机程序连接失败："+e.message);}finally{busy=false;}}
$("participant").addEventListener("change",()=>loadParticipant().catch(e=>error(e.message)));
$("model").addEventListener("change",()=>loadDetails());
$("mode").addEventListener("change",updateMode);
$("check").addEventListener("click",async()=>{try{error("");const result=await api("/api/check",requestConfig());note(result.message);}catch(e){error(e.message);}});
$("start").addEventListener("click",async()=>{error("");updateMode();$("start").disabled=true;$("result").hidden=true;resultKey="";previousCommands=-1;try{render(await api("/api/start",requestConfig()));}catch(e){error(e.message);$("start").disabled=!details;}});
$("stop").addEventListener("click",()=>api("/api/stop",{}).catch(e=>error(e.message)));
$("simulationTask").addEventListener("change",()=>{if(state?.running&&state.mode==="simulated")api("/api/simulation-task",{label:Number($("simulationTask").value)}).catch(e=>error(e.message));});
$("fullscreen").addEventListener("click",async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch(e){error("无法进入全屏，可继续在当前窗口使用。");}});
$("shutdown").addEventListener("click",async()=>{try{await api("/api/shutdown",{});polling=false;$("statusText").textContent="程序正在关闭，正在完成保存";$("start").disabled=true;$("stop").disabled=true;}catch(e){error(e.message);}});
$("copyPath").addEventListener("click",async()=>{try{await navigator.clipboard.writeText($("resultPath").value);$("copyPath").textContent="已复制";}catch(e){$("resultPath").select();error("请复制已选中的保存位置。");}});
$("importModel").addEventListener("click",async()=>{
  const button=$("importModel");try{
    error("");const file=$("modelFile").files[0];if(!file)throw new Error("请先选择模型 .pt 文件。");
    if(file.size>32*1024*1024)throw new Error("模型文件不能超过32MB。");
    const participant=$("importParticipant").value.trim();if(!participant)throw new Error("请填写模型所属被试编号。");
    button.disabled=true;button.textContent="正在导入…";
    const encoded=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(",")[1]);reader.onerror=()=>reject(new Error("读取模型文件失败"));reader.readAsDataURL(file);});
    const result=await api("/api/import-model",{participant,name:$("importName").value,data:encoded});
    await loadModels(result.participant_id,result.model_id);$("importMessage").textContent="模型已登记，已选中被试 "+result.participant_id+"。";$("importPanel").open=false;note("模型已导入并选中。可先点击“检查模型”，再开始在线解码。");
  }catch(e){error(e.message);}finally{button.disabled=!!state?.running;button.textContent="导入并登记模型";}
});
window.addEventListener("resize",()=>drawWave(state?.trace||[]));
async function init(){try{state=await api("/api/state");await loadModels(state.selection?.participant_id,state.selection?.model_id);render(state);setInterval(poll,250);}catch(e){error("无法初始化在线界面："+e.message);}}
init();
