# MaskGuardEnv

## Project Overview
MaskGuardEnv refactors the default OpenEnv scaffold into a task-specific reinforcement learning environment for policy-aware PII masking. An agent observes a document, detects sensitive entities, masks them iteratively, validates compliance, and submits only when the output satisfies the active policy.

## Environment Design
The environment keeps a single source text, tracks detected and masked entities, applies masking one entity at a time, and supports a correction loop so agents can re-check the document before submission. The core logic lives in `/Users/rnr/Documents/maskguard_openenv/env.py` and is reused by the CLI runners and FastAPI server.

## Observation Space
Each observation contains:
- `text`
- `detected_entities`
- `masked_entities`
- `remaining_entities`
- `policy_mode`
- `step_count`

## Action Space
Supported actions are defined in [/Users/rnr/Documents/maskguard_openenv/actions.py](/Users/rnr/Documents/maskguard_openenv/actions.py):
- `detect_entity`
- `mask_entity`
- `skip_entity`
- `validate_document`
- `recheck_entities`
- `submit_result`

## Reward Function
Reward shaping is implemented in [/Users/rnr/Documents/maskguard_openenv/rewards.py](/Users/rnr/Documents/maskguard_openenv/rewards.py):
- `+2` correct mask
- `-2` missed entity
- `-1` overmask
- `+5` compliance success
- `-3` invalid masking

## Policy Modes
Policy definitions live in [/Users/rnr/Documents/maskguard_openenv/policy_modes.py](/Users/rnr/Documents/maskguard_openenv/policy_modes.py):
- `GDPR`: personal contact and identity information
- `HIPAA`: health-related identifiers and IDs
- `FINANCE`: account and card details plus contact channels

## Agent Interaction Loop
1. Reset the environment with source text and a policy mode.
2. Call `detect_entity` to populate candidates.
3. Use `mask_entity` repeatedly for each remaining required entity.
4. Call `recheck_entities` to identify anything still exposed.
5. Call `validate_document` to score compliance.
6. Call `submit_result` only after validation passes.

## Evaluation Metrics
Evaluation is implemented in [/Users/rnr/Documents/maskguard_openenv/evaluator.py](/Users/rnr/Documents/maskguard_openenv/evaluator.py). The environment reports:
- precision
- recall
- F1 score
- compliance score

## Dataset Runner
The dataset runner in [/Users/rnr/Documents/maskguard_openenv/dataset_runner.py](/Users/rnr/Documents/maskguard_openenv/dataset_runner.py) loads [/Users/rnr/Documents/maskguard_openenv/datasets/sample_inputs.json](/Users/rnr/Documents/maskguard_openenv/datasets/sample_inputs.json), runs the environment across samples, and prints aggregate precision, recall, F1 score, and average reward.

## API Usage
The FastAPI app in [/Users/rnr/Documents/maskguard_openenv/server/app.py](/Users/rnr/Documents/maskguard_openenv/server/app.py) exposes:
- `POST /reset`
- `POST /step`
- `POST /submit`

Example:
```bash
uvicorn server.app:app --reload
curl -X POST http://localhost:8000/reset -H 'Content-Type: application/json' -d '{"text":"Call me at 9876543210","policy_mode":"GDPR","expected_entities":["PHONE"]}'
```

## Local Execution Steps
```bash
python inference.py
python dataset_runner.py
uvicorn server.app:app --reload
```

## Hugging Face Space Deployment
The repository still uses `openenv.yaml` as the environment manifest, so it can be packaged for OpenEnv-compatible deployments after local validation.
