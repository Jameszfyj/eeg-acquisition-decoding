# 来源说明

本项目从已有脑电采集与机械手解码工程整理，保留其采集、协议、IFNet、触发、校准和动作处理代码，并增加通用ONNX模型包接口及公开项目结构。

- `handbci/model.py`为原参考工程中的IFNet实现适配，不代表本项目提出该网络。
- `handbci/vendor/neuracle_recorder.py`及TriggerBox驱动来自原项目提供的设备示例/适配代码。
- `handbci/studio/vendor/`为已有项目的ACP协议、布局与显示实现；不是厂家官方兼容性认证。
- `handbci/vendor/inspire_serial.py`为原工程提供的机械手串口接口。
- `assets/task_cues_20261025/`是原工程中的AI生成手势提示素材，不是实测图像。

仓库不包含厂家安装器、二进制库、闭源软件文件、个人脑电或个人权重。引用第三方算法和适配器时请保留相应说明。上游作者和第三方权利不因本仓库的整理而改变。
