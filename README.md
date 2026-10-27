# EEG Acquisition & Decoding

本机运行的脑电采集与实时解码系统，包含两个独立工作台。启动网页时不会自动连接设备或驱动机械手。

| 工作台 | 功能 |
| --- | --- |
| 采集平台 | 模拟设备或 Neusen ACP 接收、阻抗检查、2～6类提示任务、TriggerBox事件核验、原始数据与质量记录、采后查看 |
| 实时解码平台 | 自己的模型导入、通道核对、连续推理、投票/稳定判决、2～6类有提示解码、可选机械手输出 |

支持 **ONNX模型包**和本项目 **IFNet格式PT**。任意PyTorch权重、Python脚本和pickle文件不能直接导入；自定义网络先导出单文件ONNX，再按输入约定打包。通道、单位、采样率、预处理及类别须与训练一致。

不包含个人脑电、个人训练权重、历史校准、录屏、运行环境或厂家安装包。原研究模型的接口代码保留，但其个人权重和训练来源不会随仓库发布。

## 快速开始

建议Windows 10/11、Python 3.10～3.12。模拟采集及模型接口测试也可在Linux运行；Windows本机中文语音和真实硬件须单独核对。

在下载的项目文件夹中执行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[decode]"
```

双击 **启动采集.cmd** 或 **启动解码.cmd**。也可使用：

```powershell
.venv\Scripts\python -m handbci.studio.server
.venv\Scripts\python -m handbci.continuous_online
```

采集工作台默认 `http://127.0.0.1:57373`，实时解码工作台默认 `http://127.0.0.1:57474`。仅监听本机。结束任务后通过页面停止并保存；关闭终端可结束服务。

## 无设备体验

1. 在采集工作台选择人工模拟设备，连接后选择模拟TriggerBox并自检，再设置任务并开始。非Windows或没有中文语音时先关闭语音。模拟数据会明确保存为人工来源。
2. 生成一个未训练的接口示例模型：

   ```powershell
   .venv\Scripts\python examples/create_demo_model.py --output runtime/demo_model.zip
   ```

3. 在解码工作台展开“导入自己的模型”，编号填写 `DEMO_ONNX`，选择上面的ZIP并导入。它会自动进入模拟脑电入口，可以开始连续解码。此模型只验证接口，不代表脑电识别效果。

## 使用自己的模型

准备 `model.onnx` 与 `manifest.json`，打包为ZIP，在网页中导入。平台检查模型节点、维度、输出概率、通道和配置后才登记；不连接设备进行导入。

完整说明：[模型接入](docs/MODEL_IMPORT.md)。支持：

- 2～6类输出；完整设备流8/16/32/64道，模型可使用命名的通道子集。
- `raw`、因果FIR及降采样、IFNet双频带三种明确的预处理方式。
- 明确的logits或概率输出，可选逐通道标准化。
- 真实ACP或明确标记的TCP人工源；人工模型不能驱动真实机械手。

也可在左侧“导入外部校准数据”中导入另一台电脑的本项目采集会话文件夹或ZIP。支持20轮、想象4秒的有标签记录，准备时长可变，按实际想象起止标记提取；至少保留配对的`session.json`和`data.h5`。检查通过后可作为原六组模型或兼容导入模型的校准数据，具体条件见[外部数据导入说明](docs/MODEL_IMPORT.md#从另一台电脑导入采集数据)。

在“真实脑电 · ACP”入口选择当前设备，核对帽配置、单位和采样率后再开始。其他品牌设备需要实现适配器，见[设备接入](docs/DEVICES.md)。本项目不会把未知硬件当成已兼容。

## 数据

采集数据位于 `runtime/studio/sessions/`；解码、模型导入和诊断位于 `runtime/` 下。每次独立建立会话，保存原始数据及配置，不覆盖已有记录。这些目录默认全部排除在Git之外。

带提示采集保留任务标签；无提示解码没有真实意图标签，不能仅凭预测日志计算准确率。软件标记是主机估计，串口ACK也不等于已核验的物理同步。原始协议、设备时间和质量限制随会话保留。

## 开发

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

测试使用人工信号、模拟串口和本机TCP，不连接实体脑电设备或机械手。GitHub Actions在Windows和Linux上执行相同测试。历史个人数据和六组模型实验不属于公开测试依赖。

代码来源与相关限制见 [NOTICE.md](NOTICE.md)；平台结构见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。
