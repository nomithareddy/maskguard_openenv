---
title: Maskguard Openenv Environment Server
emoji: 🎸
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Maskguard Openenv Environment

## Project Overview
Maskguard Openenv Environment refactors the default scaffold into a policy-aware reinforcement learning environment for PII masking. An agent observes raw text, detects sensitive entities, masks them, validates compliance, receives reward, and continues until submission is accepted.

## Environment Design
The core environment lives in [env.py](/Users/rnr/Documents/maskguard_openenv/env.py) as `MaskGuardEnv`. The existing scaffold wrapper in [server/maskguard_openenv_environment.py](/Users/rnr/Documents/maskguard_openenv/server/maskguard_openenv_environment.py) preserves the template naming convention while delegating behavior to the RL environment.

Built-in task variants:
- `contact_masking`
- `healthcare_note`
- `finance_record`

## Observation Space
Each observation contains:
- `text`
- `detected_entities`
- `masked_entities`
- `remaining_entities`
- `policy_mode`
- `step_count`

## Action Space
Supported actions are defined in [actions.py](/Users/rnr/Documents/maskguard_openenv/actions.py):
- `detect_entity`
- `mask_entity`
- `skip_entity`
- `validate_document`
- `recheck_entities`
- `submit_result`

## Reward Function
Reward shaping is implemented in [rewards.py](/Users/rnr/Documents/maskguard_openenv/rewards.py):
- `+2` correct mask
- `-2` missed entity
- `-1` overmask
- `+5` compliance success
- `-3` invalid masking

## Policy Modes
Policy definitions are implemented in [policy_modes.py](/Users/rnr/Documents/maskguard_openenv/policy_modes.py):
- `GDPR`
- `HIPAA`
- `FINANCE`

## Agent Interaction Loop
1. Reset the environment with text and a policy mode.
2. Run `detect_entity` to identify candidate PII.
3. Use `mask_entity` to replace one entity at a time.
4. Use `recheck_entities` to inspect what remains.
5. Run `validate_document` to measure compliance.
6. Run `submit_result` only when validation succeeds.

## Evaluation Metrics
Evaluation is implemented in [evaluator.py](/Users/rnr/Documents/maskguard_openenv/evaluator.py) and returns:
- precision
- recall
- F1 score
- compliance score

## Dataset Runner
The dataset runner in [dataset_runner.py](/Users/rnr/Documents/maskguard_openenv/dataset_runner.py) loads [datasets/sample_inputs.json](/Users/rnr/Documents/maskguard_openenv/datasets/sample_inputs.json), executes the environment across samples, and prints aggregate precision, recall, F1 score, and average reward.

## API Usage
The FastAPI application in [server/app.py](/Users/rnr/Documents/maskguard_openenv/server/app.py) exposes:
- `GET /health`
- `GET /state`
- `POST /reset`
- `POST /step`
- `POST /submit`

Example:
```bash
uvicorn server.app:app --reload
curl -X POST http://localhost:8000/reset -H 'Content-Type: application/json' -d '{"text":"Call me at 9876543210","policy_mode":"GDPR","target_entities":["PHONE"]}'
```

## Local Execution Steps
```bash
python inference.py
python dataset_runner.py
uvicorn server.app:app --reload
```
