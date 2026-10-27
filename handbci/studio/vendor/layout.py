"""Electrode layout + montage definitions from the vendor's Conf/setting.json.

The file is a Newtonsoft dump with "$id"/"$ref" cross references. We index every
object by $id once, then walk the pieces we need:

  ChannelSettingGroups[*].ChannelSettings   name + Theta/Radius
  Devices[*].EEGCaps[*]                     physical channel order of a cap
  <objects with Name + MontageChannels>     montage = Active/Reference weight lists

Positions are the vendor's own polar numbers, untouched:

    x = r * sin(theta)     # theta 0 = anterior, positive = right, negative = left
    y = r * cos(theta)     # positive = anterior

`head_radius` is chosen so the outermost scalp electrode sits at 92 % of the rim;
the browser scales by it. Midline electrodes without Theta land on x = 0.

An empty ChannelList named "Average" means "average of every available channel"
(that is how the vendor encodes the average reference).
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

DEFAULT_CONF = r'C:\Program Files (x86)\Neuracle\NeuroNexus\Conf\setting.json'
NON_SCALP = {'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL', 'EMG', 'A1', 'A2', 'M1', 'M2', 'REF', 'GND'}


def _resolve(obj, ids):
    return ids[obj['$ref']] if isinstance(obj, dict) and '$ref' in obj else obj


def _index_ids(node, ids):
    if isinstance(node, dict):
        if '$id' in node:
            ids[node['$id']] = node
        for v in node.values():
            _index_ids(v, ids)
    elif isinstance(node, list):
        for v in node:
            _index_ids(v, ids)


def _side(obj, ids) -> Dict:
    obj = _resolve(obj, ids) if obj else None
    if not obj:
        return {'name': '', 'terms': []}
    terms = []
    for item in obj.get('ChannelList', []):
        cs = _resolve(item.get('ChannelSetting'), ids)
        name = cs.get('Name') if cs else None
        if name:
            terms.append((name, float(item.get('Value', 1.0))))
    return {'name': obj.get('Name') or '', 'terms': terms}


def _fmt(terms: List[Tuple[str, float]]) -> str:
    if not terms:
        return ''
    parts = []
    for name, w in terms:
        if abs(w - 1.0) < 1e-9:
            parts.append(name)
        elif abs(w + 1.0) < 1e-9:
            parts.append('-' + name)
        else:
            parts.append(f'{w:g}·{name}')
    return ' + '.join(parts).replace('+ -', '− ')


class Layout:
    def __init__(self, path: str = DEFAULT_CONF, device_type: Optional[str] = None,
                 channels: int = 64, cap_name: Optional[str] = None):
        self.path = path
        self.ok = False
        self.error = ''
        self.device_type = device_type or ''
        self.channels_meta: Dict[str, Dict] = {}
        self.caps: Dict[str, List[str]] = {}
        self.device_caps: Dict[str, str] = {}
        self.device_cap_list: Dict[str, List[str]] = {}
        self.cap_options: List[Dict] = []
        self._cap_request = cap_name
        self.montages: Dict[str, List[Dict]] = {}
        self.cap_name = ''
        self.physical: List[str] = []
        self.positions: List[Dict] = []
        self.head_radius = 1.0
        if not path or not os.path.isfile(path):
            self.error = f'setting.json not found: {path}'
            return
        try:
            self._load(channels)
            self.ok = True
        except Exception as exc:  # noqa: BLE001
            self.error = f'{type(exc).__name__}: {exc}'

    # ------------------------------------------------------------------ load
    def _load(self, channels: int) -> None:
        with open(self.path, encoding='utf-8-sig') as fh:
            doc = json.load(fh)
        ids: Dict[str, dict] = {}
        _index_ids(doc, ids)

        for group in doc.get('ChannelSettingGroups', []):
            for cs in _resolve(group, ids).get('ChannelSettings', []):
                cs = _resolve(cs, ids)
                name = cs.get('Name')
                if not name:
                    continue
                self.channels_meta[name] = {
                    'name': name,
                    'theta': cs.get('Theta'),
                    'radius': cs.get('Radius'),
                    'x': cs.get('X'), 'y': cs.get('Y'), 'z': cs.get('Z'),
                }

        for dev in doc.get('Devices', []):
            dev = _resolve(dev, ids)
            dtype = dev.get('Name') or ''
            names_for_device = []
            for ref in dev.get('EEGCaps') or []:
                cap = _resolve(ref, ids)
                names = []
                for entry in cap.get('EEGCapSensorChannelSettings', []):
                    entry = _resolve(entry, ids)
                    cs = _resolve(entry.get('ChannelSetting'), ids)
                    names.append(cs.get('Name') if cs else (entry.get('Name') or ''))
                cname = cap.get('Name') or f'cap{len(self.caps)}'
                self.caps[cname] = names
                names_for_device.append(cname)
            if dtype and names_for_device:
                self.device_caps[dtype] = names_for_device[0]
                self.device_cap_list[dtype] = names_for_device

        candidates = self.device_cap_list.get(self.device_type) or list(self.caps.keys())
        self.cap_options = [{'name': c, 'count': len(self.caps[c])} for c in candidates]
        chosen = None
        if self._cap_request and self._cap_request in self.caps:
            chosen = self._cap_request              # explicit choice wins
        else:
            for name in candidates:                 # otherwise match the channel count
                if len(self.caps[name]) == channels:
                    chosen = name
                    break
        if chosen is None and candidates:
            chosen = candidates[0]
        self.cap_name = chosen or ''
        phys = list(self.caps.get(self.cap_name, []))
        if channels and len(phys) > channels:
            phys = phys[:channels]
        self.physical = phys

        self._build_positions()
        self._build_montages(ids)

    def _build_positions(self) -> None:
        pts = []
        for name in self.physical:
            meta = self.channels_meta.get(name, {})
            theta = meta.get('theta')
            r = meta.get('radius')
            if r is None:
                x0 = meta.get('y') or 0.0          # Y = lateral
                y0 = meta.get('x') or 0.0          # X = anterior
                r = math.hypot(x0, y0)
                theta = math.degrees(math.atan2(x0, y0)) if r else 0.0
            if theta is None:
                theta = 0.0                         # midline
            th = math.radians(float(theta))
            rr = float(r)
            pts.append({'name': name,
                        'theta': (float(theta) if theta else None),
                        'radius': round(rr, 4),
                        'x': round(rr * math.sin(th), 4),
                        'y': round(rr * math.cos(th), 4)})
        outer = max((math.hypot(p['x'], p['y']) for p in pts
                     if p['name'] not in NON_SCALP), default=1.0) or 1.0
        self.head_radius = round(outer / 0.92, 4)
        self.positions = pts

    def _build_montages(self, ids: Dict[str, dict]) -> None:
        self.montages['Raw'] = [{'label': name, 'active': {'name': name, 'terms': [(name, 1.0)]},
                                 'reference': {'name': '', 'terms': []}} for name in self.physical]
        for obj in ids.values():
            if 'MontageChannels' not in obj or not obj.get('Name'):
                continue
            rows = []
            for mc in obj['MontageChannels']:
                mc = _resolve(mc, ids)
                rows.append({'label': mc.get('Name') or '',
                             'active': _side(mc.get('Active'), ids),
                             'reference': _side(mc.get('Reference'), ids)})
            if rows:
                self.montages[obj['Name']] = rows

    # ------------------------------------------------------------------ api
    def channel_index(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.physical)}

    def matrix(self, montage: str) -> Tuple[List[Dict], List[str], List[str], List[str]]:
        """-> rows, labels, formulas, skipped labels. Rows for a missing electrode are dropped."""
        rows = self.montages.get(montage) or self.montages.get('Raw') or []
        idx = self.channel_index()
        n = len(self.physical)
        kept, labels, formulas, skipped = [], [], [], []
        for row in rows:
            label = row['label']
            sides = []
            ok = True
            for side in ('active', 'reference'):
                spec = row[side]
                terms = list(spec['terms'])
                if not terms and spec['name'] == 'Average':
                    terms = [(name, 1.0 / n) for name in self.physical] if n else []
                if any(name not in idx for name, _ in terms):
                    ok = False
                    break
                sides.append(terms)
            if not ok:
                skipped.append(label)
                continue
            act, ref = sides
            weights: Dict[int, float] = {}
            for name, w in act:
                weights[idx[name]] = weights.get(idx[name], 0.0) + w
            for name, w in ref:
                weights[idx[name]] = weights.get(idx[name], 0.0) - w
            weights = {k: v for k, v in weights.items() if abs(v) > 1e-9}
            if not weights:
                skipped.append(label)
                continue
            kept.append({'label': label, 'weights': {str(k): round(v, 6) for k, v in weights.items()}})
            labels.append(label)
            act_str = _fmt(act) or '0'
            ref_str = _fmt(ref)
            if row['reference']['name'] == 'Average' and not row['reference']['terms']:
                ref_str = f'avg({n})'
            formulas.append(f'{label} = {act_str}' + (f' − {ref_str}' if ref_str else ''))
        return kept, labels, formulas, skipped

    def summary(self) -> Dict:
        return {
            'ok': self.ok,
            'error': self.error,
            'source': self.path,
            'cap': self.cap_name,
            'device': self.device_type,
            'head_radius': self.head_radius,
            'caps': self.cap_options,
            'channels': self.positions,
            'montages': [{'name': n, 'count': len(r)} for n, r in self.montages.items()],
        }
