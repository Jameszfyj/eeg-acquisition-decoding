# 设备接入

## Neusen ACP

目前实现的是Neusen ACP设备和人工模拟器。电脑、放大器与无线TriggerBox须在同一局域网。ACP使用4097端口，不需要先开启厂家8712浮点转发。阻抗检查结束后再重新连接TriggerBox并核验回传。

设备必须返回与请求一致的通道数、采样率及每包点数。实时解码还要求已确认的通道名称和物理顺序。

### 通道顺序

使用以下任一种配置：

- 本机已安装厂家软件时读取其`Conf/setting.json`，或通过`STUDIO_CAP_CONFIG`指定该文件。厂家的文件不随仓库发布。
- 复制`configs/channel_layouts.json`为`configs/channel_layouts.local.json`，按实际设备输出顺序填写通道。确认后把`confirmed`改成`true`，设置`HAND_BCI_LAYOUT_FILE`指向该本地文件，再启动两个平台。

示例配置的顺序未经实机核对，默认`confirmed=false`，不会作为可靠的模型输入匹配。请按设备导出顺序填写，不要仅凭头皮位置或通道名称猜测。`positions`可选，只影响帽图显示，不改变记录顺序。

```powershell
$env:HAND_BCI_LAYOUT_FILE = (Resolve-Path configs/channel_layouts.local.json).Path
python -m handbci.studio.server
```

## TriggerBox

支持USB串口发标记及独立Wi-Fi事件接收。串口应答、事件回传和设备时钟语义是不同层次的证据。连接后必须自检；厂家型号或固件不同仍需实机验证。软件标记是主机估计，不能冒充硬件同步。

## 机械手

支持项目原有Inspire串口驱动。默认模拟输出，选择真实输出并明确开始后才连接。先确认串口、设备ID、初始姿态、动作目标、速度和实际回位；公共仓库不携带个人已确认的姿态。模型分数或指令不是动作到位证据，到位以硬件回读为准。

## 其他设备

实现`handbci.studio.devices.DeviceAdapter`：提供`connect/disconnect/impedance/describe`及`Packet`，明确`counts`、单位缩放、通道名称、设备时间和接收时间，然后通过`register_adapter`注册。当前实时平台的采集循环针对ACP；其他适配器还须接入该循环并完成时序、断流和写盘测试。不能只改端口就声称支持。
