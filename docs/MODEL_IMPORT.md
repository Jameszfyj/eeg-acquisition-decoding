# 接入自己的模型

## 两种格式

1. 本项目训练导出的 `handbci.ifnet_v2.v1` PT：带架构、配置、标准化参数和被试编号，保留原导入流程。
2. ONNX ZIP：根目录必须且只能有 `manifest.json` 和 `model.onnx`。上传最多32MB，单文件模型解压后最多128MB，不接收外部权重、Python或自定义可执行文件。

任意`.pt`不能凭文件后缀通用。自定义PyTorch、sklearn等模型须先导出ONNX，且运行所需全部变换必须包含在模型或声明的预处理流程中。没有自动猜测训练时参考、归一化、滤波或类别含义的功能。

## 生成清单模板

```python
import json
from examples.create_demo_model import build_manifest

manifest = build_manifest()
manifest["participant_id"] = "S01"
manifest["config"]["session_metadata"]["participant_id"] = "S01"
manifest["name"] = "S01 EEG classifier"
# 以下完整设备通道表必须按训练时的逻辑顺序填写。
manifest["config"]["device"].update(
    source="neuracle_acp", unit="uV", sample_rate=1000,
    stream_channels=33,
    channels=["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2",
              "FC5", "FC6", "Cz", "C3", "C4", "T7", "T8", "A1", "A2",
              "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8",
              "PO3", "PO4", "Oz", "O1", "O2"],
)
manifest["input"] = {"name": "eeg", "pipeline": "raw", "channels": ["C3", "Cz", "C4"]}
manifest["output"] = {"name": "scores", "kind": "logits"}
with open("manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
```

上面是格式示例，不能替代你的训练配置。类别ID从0连续排列，`config.classes`的顺序必须与模型输出列一致；每项包括`id/name/task_text/action`，动作尚未设置时`action`填`custom`。页面可单独选择机械手动作，不改变训练标签。

## 输入和预处理

| `input.pipeline` | 进入模型的数组 | 处理 |
| --- | --- | --- |
| `raw` | `[1,C,T]` float32 | 原始电压窗口，不滤波、不降采样；T=输入采样率×窗口秒数 |
| `fir` | `[1,C,T]` float32 | `config.preprocess`规定的因果FIR、整数倍降采样及起始剔除；T=输出采样率×窗口秒数 |
| `ifnet_filterbank` | `[1,2,C,T]` float32 | FIR后按4–16/16–40Hz双频带处理 |

`input.channels`是模型用到的子集及顺序；`config.device.channels`是完整设备流的逻辑通道表。接入后根据已核对的设备物理通道名重排，再选取子集，不按编号猜测。

可选`input.normalization={"mean":[...],"std":[...]}`，两数组长度为C，std必须为正。标准化在FIR之后、双频带之前执行。若训练采用其他方式，请在ONNX图内完成并选择合适pipeline，避免重复处理。

输出必须是 `[1,K]`，K为2～6。`logits`会经过稳定softmax；`probabilities`须非负、有限且和为1。模型只有一个float32输入和一个输出，节点名称必须与清单一致。动态批次/维度允许，但实际运行尺寸仍按清单检查。

## PyTorch导出示例

```python
model.eval()
example = torch.zeros(1, 3, 2000)  # 与训练和manifest一致
torch.onnx.export(model, example, "model.onnx", input_names=["eeg"],
                  output_names=["scores"], opset_version=17, dynamo=False)
```

然后打包：

```powershell
python -m handbci.model_bundle --manifest manifest.json --model model.onnx --output my_model.zip
```

在网页填写模型内部被试编号并上传ZIP。上传后会检查图、输入和一次试运行输出，计算校验值并登记在本机`configs/subject_models.json`；模型文件放在`runtime/models/imported/`。修改模型或清单后需重新导入。模型检查通过只证明接口可运行，不证明分类准确率。

导入模型在“按目标提示采集”模式下必须先校准，或选择一份兼容的完整校准记录。默认20轮，每轮包含当前模型的全部类别，休息2秒、想象6秒；六分类为120次任务。

校准数据保存为独立的 `calibration_raw.npz` 和 `calibration_data.json`，保留原始电压、通道顺序、任务标签、采样索引和设备信息。原研究六分类模型与导入模型都可使用这些数据。复用要求同一被试、同一设备及帽配置、任务文字一致、通道可对齐、单位与采样率一致，且轮数和时长相同；不会把旧3轮记录当成20轮记录。任务类别顺序和模型使用的通道子集可以不同。重新戴帽或调整电极后须重新校准。

每个模型重新计算自己的参数，不能共用另一模型拟合出的偏置。原六组模型继续使用原EA与动作偏置方法；导入的ONNX或本项目IFNet格式模型，按自身输入预处理回放这批脑电，再拟合有正则约束的输出logit偏置。网络权重保持原样，后续解码的目标提示不参与计算偏置。校准不保证准确率提高，也不等同于重训外部网络；若模型需要其他专属适配算法，需要单独扩展。模型窗口加滤波历史须容纳在6秒任务中。

## 从另一台电脑导入采集数据

1. 在本项目采集平台完成20轮采集；可以沿用**准备2.4秒、想象4秒**的范式。保留任务标签和已核验的任务起止事件。
2. 将单个完整会话文件夹复制到解码电脑。至少保留原始配对的 `session.json` 和 `data.h5`；也可把这份会话压成ZIP。
3. 在解码平台先选择对应被试和模型，展开左侧“导入外部校准数据”，选择文件夹或ZIP，然后点击“导入校准数据”。上传分段进行，ZIP/H5上限2GB，不通过GitHub或外部服务器。
4. 检查通过后，记录会出现在“已保存校准”列表，并自动选中。随后开始解码时才计算当前模型的校准参数及连接设备。

导入按实际任务起止事件的采样点提取**想象阶段**，不会把准备阶段当作想象数据，也不会补成6秒。准备时间不要求固定为2.4秒，每次实际准备时间可以不同；提取不使用“前一任务结束＋固定准备时长”来推算想象起点。原始时长、标签、设备信息和来源校验值随导入保存；原始JSON/H5副本保留在该记录的 `source/` 子目录。重复导入同一数据会选中已有记录。

外部采集时序可与新在线校准的6秒设置不同；仍须每类20次、至少4秒想象，且足够容纳目标模型的完整窗口及滤波历史。事件边界微小量化差异会按所有任务均具有的实际长度向内截取，不越过任务结束边界。休息及准备时长按原记录保留。原六组模型继续使用原来0.6/1.0/1.4秒起始的三个2秒窗口。

不同被试、不同任务（如把“双脚”改为“双手”）、缺少任务标签、未完整结束、通道或采样率不匹配、索引/时钟缺口和未核验的任务边界会被拒绝，页面显示原因。导入不会清除原采集的质量限制，硬件事件时间未独立测量的标记继续保留。更换或重新佩戴脑电帽后，原记录不应视作本次佩戴状态的校准。

## 本机默认设置

可在 `configs/workbench.local.json` 保存本机偏好。该文件、`local/`、`Conf/`、运行数据和环境均被Git忽略。模型仍需先导入或接入对应研究适配器；偏好不会替代模型和来源检查。

```json
{
  "format": "handbci.workbench.local.v1",
  "port": 57474,
  "default_model_id": "替换为已登记的模型编号",
  "models": {
    "替换为已登记的模型编号": {
      "hand_actions": ["index", "ring", "thumb", "little", "pinch", "middle"],
      "ui_defaults": {"decisionMode": "vote", "voteWindow": 2, "voteRatio": 0.65}
    }
  }
}
```

`hand_actions`按模型类别顺序填写，数量必须与类别数一致。默认设置仅在打开或切换模型时应用；打开页面不会启动采集或动作。

原六组有标签研究适配器额外支持 `calibration: {"rounds": 20, "imagery_seconds": 6, "rest_seconds": 2}`。每轮包含全部6类，因此20轮为120次任务。改变轮数不会重训或修改原始权重；EA及动作偏置按完整的新任务集合拟合。原来的每任务3个2秒窗口及其起始偏移0.6/1.0/1.4秒保持不变，想象提示会持续指定的全部时间。在线历史校准的轮数、时长和输入约定须一致；通过外部导入检查的4秒采集记录使用上文所述规则。

研究适配器可通过本机设置中的 `research_data_root` 指向原始六组实验目录，`labelled_checkpoint` 指向原有有标签权重，`hand_calibration` 指向已核对的机械手姿态JSON；相对路径按仓库目录解析。该入口继续校验原始模型与训练来源，依赖对应的原始文件，不会随公开仓库提供。
