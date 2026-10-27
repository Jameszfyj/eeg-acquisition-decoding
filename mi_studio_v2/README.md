# MI Studio v2 校准工具

本目录来自用户提供的六类运动想象源模型工具，现已接入本项目。仓库中的 `network.py` 保留原 IFNetRTG-L 网络、时间池化和频带操作，可在独立项目目录中加载、校准、导出模型和执行推理。

安装依赖及下载模型、数据的方法见 [模型与校准数据](../docs/BUNDLED_MODELS.md)。解压数据包到项目目录后，启动解码平台即可自动启动校准工具，也可运行：

```powershell
python -m handbci.studio_v2_runtime
```

浏览器打开 `http://127.0.0.1:57500/`。默认载入随包提供的v2源模型，选择完整原始会话目录，并按任务名称逐行确认映射，再进行源模型校准。校准完成后导入输出目录里的 `calibrated.pt`。

v2使用 `l100d250h4_pre_v2`：1000Hz FIR、250Hz高通、准备标记对齐、2秒窗、0.5秒步长。骨干和latent scaler冻结，只拟合六列逻辑回归头。不要直接把旧六类数字ID当成新输出列；详见上面的模型使用说明。

网页浏览和扫描默认限于当前项目目录。把外部采集会话复制到项目内后选择；打开工具不连接硬件。

保留的源训练代码需要完整原cargame训练环境及 `CARGAME_ROOT`，本仓库未验证完整重训流程。`REPORT_v2.md` 是随工具提供的历史技术说明，其中旧电脑路径是原说明示例，以本文件及项目文档为准。

测试使用人工数据：

```powershell
python -m pip install pytest
python -m pytest mi_studio_v2/tests -q
```
