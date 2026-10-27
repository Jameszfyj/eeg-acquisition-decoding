"""可选TriggerBox依赖、串口枚举与握手；枚举不会打开任何端口。"""
from __future__ import annotations

import ipaddress
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def validate_network(host, port):
    try:
        address = ipaddress.IPv4Address(str(host).strip())
    except ipaddress.AddressValueError as exc:
        raise ValueError("请填写TriggerBox实际的局域网IPv4地址。") from exc
    if address.is_unspecified or address.is_multicast or not (address.is_private or address.is_loopback):
        raise ValueError("TriggerBox地址须为本机或局域网IPv4地址。")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("TriggerBox网络端口须为1至65535的整数。")
    if port in {4097, 8712}:
        raise ValueError("4097是厂家设备管理接口，8712是脑电转发接口。DCP网络打标示例端口为4321，不能混用。")
    return {"mode": "network", "transport": "tcp", "host": str(address), "port": port,
            "protocol": "neuracle_dcp"}


def network_defaults():
    config_path = Path(__file__).resolve().parents[2] / "Conf/DeviceSettings.xml"
    host, model, control_port = "", "", None
    try:
        root = ET.parse(config_path).getroot()
        host = root.findtext("TriggerServerIPAddressString", "")
        model = root.findtext("TriggerTypeString", "")
        control_port = root.findtext("TriggerServerPort")
    except (OSError, ET.ParseError):
        pass
    return {"host": host, "port": 4321, "model": model, "vendor_control_port": control_port,
            "compatibility_verified": False,
            "message": f"厂家保存型号：{model or '未读取'}；IP：{host or '请手动填写'}。DCP网络打标默认4321，需读取设备名确认本型号兼容；4097不可混用。"}


def validate_port(value):
    if not isinstance(value, str) or not re.fullmatch(r"COM[1-9][0-9]{0,3}", value.strip(), re.IGNORECASE):
        raise ValueError("请选择TriggerBox实际串口，例如COM3；8712是脑电TCP端口，不是串口。")
    return value.strip().upper()


def require_serial():
    try:
        import serial
    except ImportError as exc:
        raise ValueError("缺少TriggerBox依赖pyserial，请在运行程序的Python环境安装requirements-triggerbox.txt。") from exc
    if not hasattr(serial, "Serial"):
        raise ValueError("串口模块不可用，请安装pyserial（不是名为serial的其他包）。")


def serial_ports():
    try:
        require_serial()
        from serial.tools.list_ports import comports
        ports = [{"port": p.device, "description": p.description or "串口设备"} for p in comports()]
        ports.sort(key=lambda p: (len(p["port"]), p["port"]))
        return {"available": True, "ports": ports,
                "message": "已列出本机串口，请选择TriggerBox对应端口。" if ports else "未发现串口；连接TriggerBox后点击刷新，也可手动填写COM口。"}
    except (ValueError, OSError, ImportError) as exc:
        return {"available": False, "ports": [], "message": str(exc)}


def configure_triggerbox(config, settings=None):
    """界面明确选择软件模式时清空旧串口，不静默降级硬件模式。"""
    if settings is not None:
        if not isinstance(settings, dict) or settings.get("mode") not in {"software", "hardware", "network"}:
            raise ValueError("标记方式须为software、hardware或network。")
        if settings["mode"] == "network":
            if config["device"]["source"] != "neuracle":
                raise ValueError("真实TriggerBox仅用于Neuracle设备采集；模拟源不能选择网络触发。")
            connection = validate_network(settings.get("host"), settings.get("port", 4321))
            config["device"].update(trigger_port="", trigger_connection=connection)
            return connection
        config["device"].pop("trigger_connection", None)
        port = settings.get("port") if settings["mode"] == "hardware" else ""
        if settings["mode"] == "hardware" and not port:
            raise ValueError("已选择TriggerBox，请填写或选择实际COM串口。")
    else:
        network = config["device"].get("trigger_connection")
        if network is not None:
            if not isinstance(network, dict) or network.get("mode") != "network":
                raise ValueError("TriggerBox网络配置格式无效。")
            return configure_triggerbox(config, network)
        port = config["device"].get("trigger_port", "")
    if port:
        if config["device"]["source"] != "neuracle":
            raise ValueError("真实TriggerBox仅用于Neuracle设备采集；模拟源不能选择硬件触发。")
        port = validate_port(port)
        require_serial()
    config["device"]["trigger_port"] = port or ""
    return {"mode": "hardware" if port else "software", "port": port or ""}


def open_network_triggerbox(settings):
    from .vendor.trigger_box_tcp import TcpTriggerBoxConnection
    target = validate_network(settings.get("host"), settings.get("port", 4321))
    box = TcpTriggerBoxConnection()
    try:
        box.connect(target["host"], target["port"])
        return box
    except (OSError, RuntimeError, ValueError, UnicodeError) as exc:
        box.disconnect()
        raise ValueError(f"TriggerBox网络握手失败（{target['host']}:{target['port']}）：{exc}。请确认连接设备Wi-Fi、IP地址及本型号的DCP打标接口；不会改用4097或降级软件标记。") from exc


def probe_network_triggerbox(settings):
    box = open_network_triggerbox(settings)
    try:
        return {"ok": True, "mode": "network", "host": settings["host"], "port": settings.get("port", 4321),
                "device_name": box.device_name, "eeg_verified": False,
                "message": f"网络DCP握手通过，设备：{box.device_name}。尚未验证EEG触发；正式采集逐条核验，测试结束释放连接。"}
    finally:
        box.disconnect()


def probe_triggerbox(port):
    """只读取设备名，不发送实验事件，也不宣称EEG输入链路已通过。"""
    port = validate_port(port)
    require_serial()
    from .vendor.trigger_box import TriggerBoxConnection
    box = TriggerBoxConnection()
    try:
        box.connect(port)
        return {"ok": True, "port": port, "device_name": box.device_name,
                "eeg_verified": False,
                "message": f"{port} 串口握手通过，设备：{box.device_name or '未提供名称'}。尚未验证触发进入脑电；正式采集会逐条检查。"}
    except (OSError, RuntimeError, TimeoutError) as exc:
        raise ValueError(f"TriggerBox连接失败（{port}）：{exc}。请核对串口、驱动及是否被其他程序占用。") from exc
    finally:
        box.disconnect()
