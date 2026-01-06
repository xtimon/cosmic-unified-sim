# Unified Cosmological Simulation Results

> **Generation Date:** January 2026  
> **Framework Version:** 0.2.0

## 📁 Directory Structure

```
simulation-results/
├── quantum/           # Quantum simulations
├── cosmic/            # N-body gravitational simulations
├── coherence/         # Universe coherence evolution
├── genesis/           # Leptogenesis and matter genesis
├── holographic/       # Holographic principle
├── simulation_results.json   # Complete numerical data
├── analysis_report.md        # Automatic report
└── README.md                 # This file
```

---

## 🔬 1. Quantum Simulations (`quantum/`)

### Files
- `quantum_state_distribution.png` — probability distribution of entangled state
- `entanglement_strength_analysis.png` — entropy dependence on entanglement strength
- `observer_decoherence.png` — decoherence during observation

### Main Results

| System | Entanglement Entropy | Non-zero States |
|--------|---------------------|-----------------|
| 2 qubits | 1.000 | 2 |
| 3 qubits | 1.000 | 4 |
| 4 qubits | 1.000 | 8 |
| 5 qubits | 1.000 | 16 |

### Conclusions

1. **Maximum entanglement** is achieved in all configurations — von Neumann entropy is close to 1.0
2. **Measurement statistics** match quantum mechanics: 520/480 at 1000 measurements (within 1.3σ)
3. **Decoherence** during observation occurs at timescales ~10⁻⁵ sec for a human observer

---

## 🌍 2. N-body Simulations (`cosmic/`)

### Files
- `earth_moon_trajectory.png` — Moon's orbit (27.3 days)
- `inner_solar_system.png` — inner Solar System (1 year)
- `full_solar_system.png` — full Solar System (5 years)
- `three_body_figure8.png` — figure-8 (three-body problem)
- `binary_star_system.png` — binary star (10 years)

### Energy Conservation

| System | Duration | Energy Change |
|--------|----------|---------------|
| Earth-Moon | 27.3 days | **0.000000%** |
| Inner Solar System | 1 year | **8.4×10⁻⁷%** |
| Full Solar System | 5 years | **1.7×10⁻⁷%** |
| Figure-8 | 10⁷ sec | **1.3×10⁻⁶%** |
| Binary Star | 10 years | **0.000001%** |

### Conclusions

1. **RK45 integrator** provides excellent energy conservation (<10⁻⁶%)
2. **Solar System** is stable on year timescales with high accuracy
3. **Figure-8** — periodic solution of the three-body problem successfully reproduced
4. **Bug fixed** in orbital velocity formula for binary stars (was 10¹²²%, now 10⁻⁶%)

---

## 🌌 3. Coherence Evolution (`coherence/`)

### Files
- `coherence_evolution_standard.png` — standard K(n) evolution
- `coherence_model_comparison.png` — model comparison
- `coherence_alpha_sensitivity.png` — sensitivity to parameter α
- `coherence_future_prediction.png` — future evolution prediction
- `symmetry_breaking_potential.png` — spontaneous symmetry breaking

### Model Comparison

| Model | Growth Factor K(12)/K(0) |
|-------|-------------------------|
| Standard | **3.60×** |
| Corrected | 1.20× |
| Quantum | 1.08× |
| With Dark Energy | 2418× |

### Information Analysis

- **Shannon Entropy:** 3.46 bits
- **Maximum Entropy:** 3.58 bits
- **Efficiency:** 96.6%

### Conclusions

1. **Standard model** gives coherence growth of ~3.6 times over 12 stages
2. **Dark energy model** gives unphysical exponential growth
3. **Sensitivity to α**: at α = 0.3→0.9, growth factor changes 1.84→5.49
4. **Information efficiency** is close to optimal (96.6%)

---

## ⚛️ 4. Leptogenesis and Matter Genesis (`genesis/`)

### Files
- `parametric_resonance.png` — parametric resonance
- `leptogenesis_asymmetry.png` — baryon asymmetry
- `matter_genesis_evolution.png` — early Universe evolution
- `leptogenesis_convergence.png` — convergence analysis
- `leptogenesis_detailed_scan.png` — parameter scan
- `precise_leptogenesis_match.png` — precise match with observations
- And others...

### Leptogenesis: Key Results

**Observed value:** η_B = 6.1×10⁻¹⁰

**Optimal parameters to reproduce observations:**

| Parameter | Value |
|-----------|-------|
| Heavy neutrino mass M | **1.43×10¹¹ GeV** |
| CP violation ε | **7.0×10⁻⁷** |
| Yukawa coupling | 10⁻⁷ |
| Result η_B | 6.15×10⁻¹⁰ |
| **Error** | **0.88%** |

### Important Discovery: Convergence

For correct results, integration up to **z = M/T ≥ 5000** is necessary:

| z_max | η_B | Change |
|-------|-----|--------|
| 100 | 7.4×10⁻¹¹ | — |
| 500 | 1.9×10⁻⁹ | +2500% |
| 2000 | 3.0×10⁻⁹ | +58% |
| 5000 | 3.0×10⁻⁹ | **0%** ✓ |

### Conclusions

1. **Leptogenesis reproduces** the observed baryon asymmetry with accuracy <1%
2. **Optimal neutrino mass:** M ~ 10¹¹ GeV (GUT scale)
3. **Critical**: integration up to z ≥ 5000 for result convergence
4. **Dependence:** η_B ∝ ε/M (diagonal on parameter map)

---

## 🔮 5. Holographic Analysis (`holographic/`)

### Files
- `holographic_k_models.png` — k parameter for different models
- `holographic_formula_comparison.png` — formula comparison
- `holographic_summary.png` — summary of key relationships

### Ratio k/α ≈ 66

| Model | k | k/α |
|-------|---|-----|
| Planck 2018 | 0.4747 | 65.06 |
| WMAP9 | 0.4757 | 65.19 |
| SH0ES | 0.4747 | 65.06 |
| DES | 0.4747 | 65.06 |
| ACT | 0.4747 | 65.06 |
| **Average** | **0.4749** | **65.08** |
| **Theor. 66α** | 0.4816 | 66.00 |

### Formula Comparison for k

| Formula | k | Error vs observations |
|---------|---|----------------------|
| Observed | 0.4837 | — |
| **Boson mass** | **0.4831** | **0.12%** ✓ |
| Entropic | 0.4816 | 0.42% |
| Holographic | 0.4747 | 1.85% |
| Dark energy | 0.1495 | 69% |

### Statistical Significance

- **p-value:** 0.0233
- **Significant at α = 0.05:** ✅ Yes

### Conclusions

1. **Ratio k/α ≈ 66** is statistically significant (p = 0.023)
2. **Best formula** — based on boson mass (error 0.12%)
3. **Information capacity** of the Universe: ~10¹²³ bits

---

## 🎯 Main Scientific Conclusions

### 1. Quantum Mechanics
✅ Simulator correctly reproduces entanglement and decoherence

### 2. Gravitational Dynamics  
✅ N-body integrator is stable with energy conservation <10⁻⁶%

### 3. Cosmological Evolution
✅ Coherence model shows ~3.6× growth over the history of the Universe

### 4. Baryogenesis
✅ **Leptogenesis explains the origin of matter:**
- M ~ 10¹¹ GeV, ε ~ 10⁻⁶
- Match with η_B = 6.1×10⁻¹⁰ with accuracy <1%

### 5. Fundamental Relationships
✅ **k/α ≈ 66** — possible connection between the fine structure constant and cosmological parameters

---

## 🔄 Reproducing Results

```bash
# Install package from PyPI
pip install cosmic-unified-sim

# Or for development
git clone https://github.com/xtimon/unified-sim.git
cd unified-sim
pip install -e .

# Run all simulations
python run_all_simulations.py

# Or individual modules via CLI
sim quantum --qubits 5 --entangle
sim cosmic --system solar --days 365
sim coherence --stages 12 --alpha 0.66
```

---

## 📚 References

- [Full Documentation](https://cosmic-unified-sim.readthedocs.io/)
- [PyPI Package](https://pypi.org/project/cosmic-unified-sim/)
- [GitHub Repository](https://github.com/xtimon/unified-sim)
- [Source Code](../sim/)
- [Jupyter Examples](../examples/)

---

*Generated by Unified Cosmological Simulation Framework v0.2.0*
