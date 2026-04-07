---
title: Maskguard Openenv Environment Server
emoji: 🎸
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Maskguard Openenv Environment

## Project Overview
Maskguard Openenv Environment is a real-world reinforcement learning environment for policy-aware PII masking. An agent observes document text, detects sensitive entities, masks them iteratively, validates compliance, receives normalized reward in the range `0.0` to `1.0`, and submits a final compliant result.

## Motivation
Real organizations must redact sensitive data before documents can be stored, shared, or used for downstream AI workflows. This environment models that production-style problem as an RL task, so agents must make sequential masking decisions, recover from missed entities, and optimize toward compliant document handling rather than solving a toy benchmark.

## Environment Design
The core environment lives in [env.py](/Users/rnr/Documents/maskguard_openenv/env.py) as `MaskGuardEnv`. The scaffold wrapper in [server/maskguard_openenv_environment.py](/Users/rnr/Documents/maskguard_openenv/server/maskguard_openenv_environment.py) preserves the template naming convention while delegating behavior to the RL environment. The environment implements `reset()`, `step()`, `validate()`, `submit()`, and `state()`.

Built-in task variants with agent graders:
- `contact_masking`: `easy`, graded by `contact_masking_grader`
- `healthcare_note`: `medium`, graded by `healthcare_note_grader`
- `finance_record`: `hard`, graded by `finance_record_grader`, with multi-entity financial records, distractor references, and exploit-aware validation

## Observation Space
Each observation contains:
- `text`
- `detected_entities`
- `masked_entities`
- `remaining_entities`
- `policy_mode`
- `step_count`
- `task_name`
- `difficulty`
- `score`

## Action Space
Supported actions are defined in [actions.py](/Users/rnr/Documents/maskguard_openenv/actions.py):
- `detect_entity`
- `mask_entity`
- `skip_entity`
- `validate_document`
- `recheck_entities`
- `submit_result`

## Reward Function
Reward shaping is implemented in [rewards.py](/Users/rnr/Documents/maskguard_openenv/rewards.py). Raw shaping captures partial progress, and external rewards are normalized to `0.0` to `1.0`:
- correct mask: positive signal
- missed entity: negative signal
- overmask: negative signal
- compliance success: positive signal
- invalid masking: negative signal

Explicit raw reward formula:
```text
raw_reward = 2 * correct_masks - 2 * missed_entities - 1 * overmasks - 3 * invalid_masks + 5 * compliance_success
normalized_reward = clamp(raw_reward, 0.0, 1.0)
```

## Policy Modes
Policy definitions are implemented in [policy_modes.py](/Users/rnr/Documents/maskguard_openenv/policy_modes.py):
- `GDPR`
- `HIPAA`
- `FINANCE`

## Agent Interaction Loop
1. Reset the environment with a built-in task or custom text.
2. Run `detect_entity` to identify candidate PII.
3. Use `mask_entity` to replace one entity at a time.
4. Use `recheck_entities` to inspect what remains.
5. Run `validate_document` to score compliance and grader output.
6. Run `submit_result` only when validation succeeds.

The hard finance task intentionally includes benign reference text that must remain visible, multiple financial identifiers, and stronger penalties for incomplete masking so graders can distinguish careful agents from brittle ones.

## Evaluation Metrics
Evaluation is implemented in [evaluator.py](/Users/rnr/Documents/maskguard_openenv/evaluator.py) and returns:
- precision
- recall
- F1 score
- compliance score
- normalized score

## Dataset Runner
The dataset runner in [dataset_runner.py](/Users/rnr/Documents/maskguard_openenv/dataset_runner.py) loads [datasets/sample_inputs.json](/Users/rnr/Documents/maskguard_openenv/datasets/sample_inputs.json), evaluates the sample dataset, and also runs the built-in `easy`, `medium`, and `hard` tasks with agent graders.

## Baseline Scores
Validated baseline results from the current deterministic inference and dataset pipeline:
- `python inference.py`
  - final score: `1.000`
  - rewards: `0.44, 0.56, 0.56, 0.75, 0.75`
- `python dataset_runner.py`
  - precision: `1.000`
  - recall: `1.000`
  - F1 score: `1.000`
  - average reward: `3.083`
  - average task score: `1.000`

## API Usage
The FastAPI application in [server/app.py](/Users/rnr/Documents/maskguard_openenv/server/app.py) exposes:
- `GET /health`
- `GET /state`
- `POST /reset`
- `POST /step`
- `POST /submit`

### Minimal Inputs
For normal testing, you only need a very small subset of fields.

Use `POST /reset` with:
- `task_name`: `contact_masking`, `healthcare_note`, or `finance_record`
- `policy_mode`: `GDPR`, `HIPAA`, or `FINANCE`

Use `POST /step` with:
- `detect_entity`: only `action_type`
- `mask_entity`: `action_type` and `entity_id`
- `validate_document`: only `action_type`
- `submit_result`: only `action_type`

These fields are optional advanced fields and can usually be left empty:
- `entity_type`
- `entity_value`
- `text`
- `target_entities`

### Copy-Paste Examples
Reset an easy task:
```bash
uvicorn server.app:app --reload
curl -X POST http://localhost:8000/reset -H 'Content-Type: application/json' -d '{"task_name":"contact_masking","policy_mode":"GDPR"}'
```

Detect entities:
```bash
curl -X POST http://localhost:8000/step -H 'Content-Type: application/json' -d '{"action_type":"detect_entity"}'
```

Mask one entity:
```bash
curl -X POST http://localhost:8000/step -H 'Content-Type: application/json' -d '{"action_type":"mask_entity","entity_id":"EMAIL-12-0"}'
```

Validate:
```bash
curl -X POST http://localhost:8000/step -H 'Content-Type: application/json' -d '{"action_type":"validate_document"}'
```

Submit:
```bash
curl -X POST http://localhost:8000/submit -H 'Content-Type: application/json' -d '{}'
```

## Local Execution Steps
```bash
uv sync
source .venv/bin/activate
python inference.py
python dataset_runner.py
uvicorn server.app:app --reload
```
