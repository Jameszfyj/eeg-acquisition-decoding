"""Class scores frozen at decision time, associated with requested/read-back actions."""
from __future__ import annotations

import copy
import math


def _probability(value):
    if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1:
        return float(value)
    return None


def build_class_evidence(classes, decision, probabilities):
    voting = decision.get('decision_mode') == 'vote'
    values = decision.get('mean_probabilities', []) if voting else probabilities
    values = values if isinstance(values, list) and len(values) == len(classes) else [None] * len(classes)
    counts, total = decision.get('vote_counts', []), decision.get('vote_total')
    counted = (voting and isinstance(counts, list) and len(counts) == len(classes)
               and type(total) is int and total > 0)
    rows = []
    for i, cls in enumerate(classes):
        probability = _probability(values[i])
        # Old records know only the winning class probability.
        if probability is None and i == decision['class_id']:
            probability = _probability(decision.get('confidence'))
        count = counts[i] if counted and type(counts[i]) is int and 0 <= counts[i] <= total else None
        rows.append(dict(class_id=cls['id'], hand_action=cls.get('hand_action'),
            task=cls.get('task_text', cls['name']), probability=probability,
            vote_share=count / total if count is not None else None))
    return rows


def instruction_evidence(command):
    label = command.get('target_label', command.get('model_class_id'))
    for row in command.get('class_evidence', []):
        if row['class_id'] == label:
            return copy.deepcopy(row)
    same_class = label == command.get('model_class_id')
    return dict(class_id=label, hand_action=command.get('target_hand_action') or command.get('hand_action'),
        task=command.get('target_task') or (command.get('task') if same_class else None),
        probability=_probability(command.get('confidence')) if same_class else None,
        vote_share=_probability(command.get('vote_share')) if same_class else None)


def actual_evidence(command, feedback):
    if (command.get('physical_action') is not True or feedback.get('source') != 'joint_position_readback'
            or feedback.get('target_confirmed') is not True):
        return None
    key = feedback.get('actual_hand_action')
    if key is None:
        return None
    rows = [row for row in command.get('class_evidence', []) if row.get('hand_action') == key]
    for row in rows:
        if row['class_id'] == command.get('model_class_id'):
            return copy.deepcopy(row)
    if len(rows) == 1:
        return copy.deepcopy(rows[0])
    if not rows and key == command.get('hand_action'):
        return dict(class_id=command.get('model_class_id'), hand_action=key, task=command.get('task'),
            probability=_probability(command.get('confidence')), vote_share=_probability(command.get('vote_share')))
    # A physical pose alone cannot identify which of multiple mapped MI classes was intended.
    return dict(class_id=None, hand_action=key, task=' / '.join(row['task'] for row in rows) or None,
                probability=None, vote_share=None)
