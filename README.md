# Quantum Traffic Gridlock — Team QuantMaxxing

**Event:** QC@IITI × IBM Qiskit Fall Fest 2026  
**Team:** QuantMaxxing · **Member:** Hussien Sulyman  
**Email:** hussiensulyman@gmail.com  
**Tag:** `v1.0-final-submit`

A scenario-independent fluid-surrogate QUBO and QAOA pipeline that schedules traffic signals across two linked intersections (A → B). The QUBO uses a half-duty reference fluid to cap effective service capacity, preventing the surrogate from "serving" vehicles that aren't in the queue. The QAOA is trained on an exact NumPy statevector and sampled with 4,096 shots on Qiskit Aer.

## Key Results

| Scenario | True Optimum C* | QUBO Minimizer | Surrogate Regret |
| :--- | :---: | :---: | :---: |
| **R** (public) | 72.0 | 72.0 | **0.00%** |
| **H** (evaluation) | 234.0 | 236.0 | **0.85%** |

QAOA sampling quality on R (5 seeds, p=3): min lift **0.614**, mean elite enrichment **6.8**.

## Repository Structure

- `run.py` — Main pipeline (executable)
- `surr.py` — Fluid-surrogate QUBO generator
- `baselines.py` — Baseline calculations (alternating, greedy, enumeration)
- `check.py` — Consistency checker (NumPy model vs Qiskit circuit)
- `exp.py` — Experiment utilities (random scenario generation)
- `bench.py` — Benchmarking script
- `scorer.py` — Official evaluator (from organizers)
- `scenario_R.json` — Public scenario R
- `scenario_H.json` — Evaluation scenario H
- `requirements.txt` — Python dependencies
- `traffic_report.pdf` — Technical report (3 pages)
- `traffic_ai_disclosure.pdf` — AI-use disclosure (1 page)
- `REFERENCE_README.md` — Organizer's reference README
- `out_0/` — Outputs for scenario R, seed 0 (qubo.json, circuit.qasm, counts.json, schedule.json)
- `out_H_0/` — Outputs for scenario H, seed 0 (qubo.json, circuit.qasm, counts.json, schedule.json)

## Installation

pip install -r requirements.txt

**Dependencies:**
- qiskit>=1.0.0
- qiskit-aer>=0.14.0
- numpy>=1.24.0
- scipy>=1.10.0

## Running the Pipeline

python run.py --scenario scenario_R.json --seed 0 --out out_0
python run.py --scenario scenario_H.json --seed 0 --out out_H_0

The pipeline produces four files in the output directory:
- `qubo.json` — QUBO matrix, Ising form, variable mapping, QAOA parameters
- `circuit.qasm` — Final bound-parameter QAOA circuit in OpenQASM 3
- `counts.json` — Exactly 4,096 raw shots as bitstring → count
- `schedule.json` — Decoded 8-bit schedule with bit order documentation

## Running Baselines and Checks

# Compute baselines (alternating, greedy, enumeration, QUBO minimizer)
python baselines.py scenario_R.json

# Verify consistency between NumPy model and Qiskit circuit
python check.py

## Mathematical Approach

The QUBO is derived from a **fluid surrogate** where queues evolve affinely:

$$q_t = q_{t-1} + a_t - g \cdot \hat{c}_t$$

$$\hat{c}_t = \min\!\big(C, r_{t-1} + a_t\big)$$

with r being a half-duty reference fluid backlog. The overflow penalty uses a convex quadratic proxy (k·w/S)·q² with global constant k = 1.5. No auxiliary variables are required — the QUBO is exactly 8×8.

## Reports

- **Technical report:** `traffic_report.pdf` (modelling, QUBO derivation, QAOA design, baselines, ablations, limitations)
- **AI-use disclosure:** `traffic_ai_disclosure.pdf`

## License
This work was developed during the QC@IITI × IBM Qiskit Fall Fest 2026 hackathon.