# Quantum Traffic Gridlock — QUBO + QAOA pipeline
Install: `pip install -r requirements.txt` (Python 3.10+).
Run: `python run.py --scenario scenario_R.json --seed 0 --out outputs` -> qubo.json, circuit.qasm, counts.json, schedule.json
Bit order: A slots 1-4 then B slots 1-4; 1 = NS green, 0 = EW green; counts.json key char i = schedule bit i.
Options: `--depth p` (default 3), `--k_ov` overflow-proxy scale (default 1.5), `--init ramp|random|zero`.
Evaluate: `python bench.py` (5 seeds: lift, elite enrichment), `python baselines.py scenario_R.json`, `python ablate.py`.
Model (see docs/traffic_report.pdf): fluid surrogate, affine queues, effective-capacity cap, quadratic overflow proxy, exact switches, 8 qubits, no auxiliary variables.
Stored results for R (seeds 0-4): outputs_R/. Scenario H: run the same command once released and add the outputs.
