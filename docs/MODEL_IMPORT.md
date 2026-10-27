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

原三轮校准属于本项目特定的IFNet与复判方案，不会自动套到任意ONNX模型。需要校准的外部模型应先完成自己的校准并导出对应模型，或扩展适配器。

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

原六组有标签研究适配器额外支持 `calibration: {"rounds": 20, "imagery_seconds": 6, "rest_seconds": 3}`。每轮包含全部6类，因此20轮为120次任务。改变轮数不会重训或修改原始权重；EA及动作偏置按完整的新任务集合拟合。原来的每任务3个2秒窗口及其起始偏移0.6/1.0/1.4秒保持不变，想象提示会持续指定的全部时间。历史校准只有轮数、时长和输入约定一致时才能复用。

研究适配器可通过本机设置中的 `research_data_root` 指向原始六组实验目录，`labelled_checkpoint` 指向原有有标签权重，`hand_calibration` 指向已核对的机械手姿态JSON；相对路径按仓库目录解析。该入口继续校验原始模型与训练来源，依赖对应的原始文件，不会随公开仓库提供。
