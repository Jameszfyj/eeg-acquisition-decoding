"use strict";
const $=id=>document.getElementById(id);
const token=document.querySelector('meta[name="viewer-token"]').content;
const colors=["#167a9d","#7058a4","#c07825","#248666","#bb5b83","#546ab4"];
let sessions=[],info=null,windowData=null,requestId=0;
const sourceName=s=>s==="neuracle"?"真实设备记录":s==="synthetic"||s==="synthetic_tcp"?"人工模拟记录":"未知来源";
function error(text=""){$("error").textContent=text;$("error").hidden=!text;}
async function api(path,options){const r=await fetch(path,options);const value=await r.json();if(!r.ok)throw new Error(value.error||"读取失败");return value;}
function busy(value){for(const id of ["draw","whole","next","previous"])$(id).disabled=value;}
function unitText(unit){return unit==="synthetic_uV"?"模拟微伏（µV）":unit==="device_native"?"设备原始单位（待核实）":unit==="uV"||unit==="µV"?"微伏（µV）":unit||"未注明单位";}
function selectedChannels(){return [...document.querySelectorAll("#channelList input:checked")].map(x=>Number(x.value));}
async function refresh(){
  error();const previous=$("session").value,participant=$("participant").value;
  const result=await api("/api/sessions");sessions=result.sessions;
  $("participant").replaceChildren(new Option("全部被试",""));
  [...new Set(sessions.map(s=>s.participant))].sort().forEach(p=>$("participant").add(new Option(p,p)));
  if([...$("participant").options].some(x=>x.value===participant))$("participant").value=participant;
  $("empty").textContent=sessions.length?"":"没有已结束的采集记录。请先完成一次采集，再刷新。仅支持本项目保存的session。";
  $("empty").hidden=!!sessions.length;
  await fillSessions(previous);
}
async function fillSessions(previous=""){
  const participant=$("participant").value;
  const rows=sessions.filter(row=>!participant||row.participant===participant);
  $("session").replaceChildren();
  rows.forEach(row=>$("session").add(new Option(row.name+" · "+row.participant+" · "+sourceName(row.source)+(row.complete?"":" · 未完成"),row.id)));
  if(rows.some(x=>x.id===previous))$("session").value=previous;
  $("record").hidden=!rows.length;
  if(rows.length)await loadSession();
}
async function loadSession(){
  const version=++requestId;error();busy(true);$("status").textContent="正在读取记录…";info=null;windowData=null;$("record").hidden=true;
  try{
    const next=await api("/api/session?id="+encodeURIComponent($("session").value));
    if(version!==requestId)return;
    info=next;windowData=null;$("record").hidden=false;
    const simulated=next.source==="synthetic"||next.source==="synthetic_tcp";
    $("notice").className="notice"+(simulated||!next.complete?" warn":"");
    $("notice").textContent=(simulated?"此记录全部为人工模拟，不是真实脑电。":"此文件记录的数据来源为真实设备。")+
      (next.complete?"采集流程已完整结束；信号质量仍需另行检查。":"本次采集未完整结束，仅供检查已保存的片段；当前训练入口会拒绝直接使用。")+
      (next.error?" 原因："+next.error:"");
    $("source").textContent=sourceName(next.source);$("duration").textContent=next.duration.toFixed(2)+" 秒";
    $("channelsInfo").textContent=next.channels.length+" 路 / "+next.sample_rate+" Hz";
    $("complete").textContent=next.complete?"完整结束":"未完成";
    $("path").value=next.path;
    $("unit").textContent="幅值："+unitText(next.unit);
    const markings=next.marking_types.map(x=>x==="hardware_trigger"?"硬件事件标记":x==="software_receive_clock"?"软件近似标记":x==="synthetic_tcp_trigger"?"人工模拟事件标记":x).join("、");
    $("marking").textContent="标记方式："+(markings||"无任务标记")+"。时间依据文件保存的采样率换算；查看器不自动核验实际采样率。";
    $("channelList").replaceChildren();
    next.channels.forEach((name,i)=>{const label=document.createElement("label"),input=document.createElement("input");input.type="checkbox";input.value=i;input.checked=i<Math.min(4,next.channels.length);input.addEventListener("change",()=>{if(selectedChannels().length>8){input.checked=false;error("每次最多显示8个通道，可分批查看。");}updateChannelCount();});label.append(input,document.createTextNode(name));$("channelList").append(label);});
    updateChannelCount();$("trial").replaceChildren(new Option("自选时间区间",""));
    next.trials.forEach((trial,i)=>$("trial").add(new Option("Trial "+trial.id+" · "+trial.task+(trial.complete?"":"（未结束）"),i)));
    $("legend").replaceChildren();
    next.classes.forEach(c=>{const item=document.createElement("span");item.style.borderColor=colors[c.id%colors.length];item.textContent=c.task_text;$("legend").append(item);});
    $("start").value="0";$("end").value=Math.min(10,next.duration).toFixed(3);
    $("start").max=next.duration;$("end").max=next.duration;
    await drawWindow();
  }catch(e){if(version===requestId){error(e.message);$("status").textContent="读取失败";}}
  finally{if(version===requestId)busy(false);}
}
function updateChannelCount(){$("channelCount").textContent="（已选 "+selectedChannels().length+" 路）";}
async function drawWindow(){
  if(!info)return;
  const selected=selectedChannels();
  if(!selected.length){error("请至少选择一个通道。");return;}
  const start=Number($("start").value),end=Number($("end").value),identity=info.id;
  if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||start>=info.duration){error("请填写有效的开始、结束时间。");return;}
  const version=++requestId;busy(true);error();$("status").textContent="正在读取波形…";
  try{
    const args=new URLSearchParams({id:identity,channels:selected.join(","),start:String(start),end:String(end)});
    const value=await api("/api/window?"+args.toString());
    if(version!==requestId||info?.id!==identity)return;
    windowData=value;$("end").value=value.end.toFixed(3);
    $("status").textContent="已显示 "+value.start.toFixed(2)+"–"+value.end.toFixed(2)+" 秒，共 "+value.n_samples.toLocaleString()+" 个原始采样点/通道";
    $("compression").textContent=(value.compressed?"当前为保留极值的波形概览；缩短时间区间可看更细的原始波形。":"当前区间直接显示原始采样点。")+
      (value.gaps?" 此区间发现采样索引断段，断点不连线，请复核记录。":"")+
      (value.traces.some(t=>t.invalid_samples)?" 此区间含无效数值，已留空显示。":"");
    paint();
  }catch(e){if(version===requestId){error(e.message);$("status").textContent="未更新波形";}}
  finally{if(version===requestId)busy(false);}
}
function paint(){
  if(!windowData)return;
  const data=windowData,canvas=$("waveform"),width=canvas.getBoundingClientRect().width,height=Math.max(260,data.traces.length*133+35),ratio=window.devicePixelRatio||1;
  canvas.style.height=height+"px";canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
  const ctx=canvas.getContext("2d");ctx.scale(ratio,ratio);ctx.clearRect(0,0,width,height);
  const left=68,right=13,top=15,bottom=30,pw=width-left-right,rowHeight=(height-top-bottom)/data.traces.length;
  const x=t=>left+(t-data.start)/(data.end-data.start)*pw;
  ctx.font='11px "Microsoft YaHei", sans-serif';
  data.traces.forEach((trace,index)=>{
    const y0=top+index*rowHeight,pad=19;
    const finite=trace.points.filter(p=>p!==null).map(p=>p[1]);
    let low=finite.length?Math.min(...finite):-1,high=finite.length?Math.max(...finite):1;
    const margin=high===low?Math.max(1,Math.abs(high)*.1):(high-low)*.12;low-=margin;high+=margin;
    const y=v=>y0+pad+(high-v)/(high-low)*(rowHeight-pad*2);
    ctx.save();ctx.beginPath();ctx.rect(left,y0,pw,rowHeight);ctx.clip();
    data.trials.forEach(trial=>{const end=trial.end??data.end;ctx.fillStyle=colors[(trial.label??0)%colors.length]+"20";ctx.fillRect(x(Math.max(data.start,trial.start)),y0,x(Math.min(data.end,end))-x(Math.max(data.start,trial.start)),rowHeight-1);});
    ctx.strokeStyle="#e6edf0";ctx.lineWidth=1;
    for(let tick=0;tick<=5;tick++){const xx=left+pw*tick/5;ctx.beginPath();ctx.moveTo(xx,y0);ctx.lineTo(xx,y0+rowHeight);ctx.stroke();}
    ctx.strokeStyle="#a6bdc7";ctx.beginPath();ctx.moveTo(left,y0+rowHeight-1);ctx.lineTo(width-right,y0+rowHeight-1);ctx.stroke();
    ctx.strokeStyle="#245c86";ctx.lineWidth=.9;ctx.beginPath();let active=false;
    for(const point of trace.points){if(point===null){active=false;continue;}if(active)ctx.lineTo(x(point[0]),y(point[1]));else{ctx.moveTo(x(point[0]),y(point[1]));active=true;}}ctx.stroke();ctx.restore();
    ctx.fillStyle="#355867";ctx.textAlign="left";ctx.fillText(trace.name,1,y0+18);
    ctx.fillStyle="#7a9099";ctx.textAlign="right";ctx.fillText(high.toPrecision(3),left-7,y0+33);ctx.fillText(low.toPrecision(3),left-7,y0+rowHeight-12);
  });
  ctx.textAlign="center";ctx.fillStyle="#728791";for(let tick=0;tick<=5;tick++)ctx.fillText((data.start+(data.end-data.start)*tick/5).toFixed(2),left+pw*tick/5,height-13);
}
function shift(direction){if(!info||!windowData)return;const span=windowData.end-windowData.start,start=Math.max(0,Math.min(info.duration-span,windowData.start+direction*span));$("trial").value="";$("start").value=start.toFixed(3);$("end").value=Math.min(info.duration,start+span).toFixed(3);drawWindow();}
$("participant").addEventListener("change",()=>fillSessions().catch(e=>error(e.message)));
$("session").addEventListener("change",loadSession);
$("refresh").addEventListener("click",()=>refresh().catch(e=>error(e.message)));
$("draw").addEventListener("click",()=>{$("trial").value="";drawWindow();});
$("whole").addEventListener("click",()=>{if(info){$("trial").value="";$("start").value=0;$("end").value=info.duration;drawWindow();}});
$("trial").addEventListener("change",()=>{if(!info||$("trial").value==="")return;const trial=info.trials[Number($("trial").value)];$("start").value=Math.max(0,trial.start-.5).toFixed(3);$("end").value=Math.min(info.duration,(trial.end??Math.min(info.duration,trial.start+4))+.5).toFixed(3);drawWindow();});
$("previous").addEventListener("click",()=>shift(-1));$("next").addEventListener("click",()=>shift(1));
$("copy").addEventListener("click",async()=>{try{await navigator.clipboard.writeText($("path").value);$("copy").textContent="已复制";}catch(e){$("path").select();error("请复制已选中的文件夹位置。");}});
$("shutdown").addEventListener("click",async()=>{try{await api("/api/shutdown",{method:"POST",headers:{"X-Viewer-Token":token}});busy(true);$("shutdown").disabled=true;$("status").textContent="查看器已关闭，原始文件保留。";}catch(e){error(e.message);}});
window.addEventListener("resize",paint);
refresh().catch(e=>error(e.message));
