#!/usr/bin/env python3
"""
Quick Start Examples
====================

Demonstrates basic usage of the unified simulation framework.
"""

import numpy as np


def quantum_example():
    """Quantum mechanics example."""
    print("=" * 60)
    print("QUANTUM SIMULATION")
    print("=" * 60)
    
    from sim.quantum import QuantumFabric, EmergentLaws, HUMAN_OBSERVER
    
    # Create 3-qubit system
    qf = QuantumFabric(num_qubits=3)
    print(f"Initial state: {qf.get_state_info()}")
    
    # Create entanglement (Bell states)
    qf.apply_entanglement_operator([(0, 1), (1, 2)])
    print(f"After entanglement: {qf.get_state_info()}")
    print(f"Entanglement entropy: {qf.get_entanglement_entropy():.4f}")
    
    # Measure a qubit
    result = qf.measure(0)
    print(f"Measurement of qubit 0: |{result}>")
    
    # Emergent laws
    particles = EmergentLaws.simulate_particle_creation(vacuum_energy=0.2)
    print(f"\nParticle creation: {len(particles)} pairs created")
    
    # Landauer principle
    energy = EmergentLaws.landauer_principle(bits_erased=1e6, temperature=300)
    print(f"Landauer energy for 10^6 bits at 300K: {energy:.2e} J")
    
    # Observer decoherence
    coherence = HUMAN_OBSERVER.observe_system(1.0, observation_time=1e-10)
    print(f"Coherence after human observation (10^-10 s): {coherence:.6f}")


def cosmic_example():
    """N-body simulation example."""
    print("\n" + "=" * 60)
    print("N-BODY SIMULATION")
    print("=" * 60)
    
    from sim.cosmic import NBodySimulator, SystemPresets
    
    # Create Earth-Moon system
    presets = SystemPresets()
    bodies = presets.create_earth_moon_system()
    
    print(f"System: {[b.name for b in bodies]}")
    
    # Run simulation for 27 days (one lunar month)
    sim = NBodySimulator(bodies)
    days = 27.3
    times, states = sim.simulate(
        t_span=(0, days * 24 * 3600),
        n_points=500
    )
    
    print(f"Simulated {days} days ({len(times)} steps)")
    print(f"Total energy: {sim.get_total_energy():.4e} J")
    
    # Check energy conservation
    initial_e, change = sim.get_energy_conservation()
    print(f"Energy change: {change*100:.6f}%")


def coherence_example():
    """Coherence evolution example."""
    print("\n" + "=" * 60)
    print("COHERENCE EVOLUTION")
    print("=" * 60)
    
    from sim.coherence import CoherenceModel, UniverseSimulator
    from sim.constants import UNIVERSE_STAGES
    
    # Basic coherence evolution
    model = CoherenceModel()
    K, C, Total = model.evolve(N=12, alpha=0.66)
    
    print("Universe coherence evolution:")
    for i, stage in enumerate(UNIVERSE_STAGES):
        print(f"  {stage:20s}: K = {K[i]:.4f}")
    
    print(f"\nGrowth factor: {K[-1]/K[0]:.2f}x")
    
    # Information content
    info = model.information_content(K)
    print(f"Shannon entropy: {info['entropy']:.4f} bits")
    print(f"Efficiency: {info['efficiency']:.2%}")
    
    # Future prediction
    K_future, stages = model.predict_future(current_stage=12, total_stages=20)
    print(f"\nFuture prediction (stages 13-20):")
    for i, k in enumerate(K_future[:5]):
        print(f"  Stage {stages[i]}: K = {k:.4f}")


def genesis_example():
    """Matter genesis example."""
    print("\n" + "=" * 60)
    print("MATTER GENESIS")
    print("=" * 60)
    
    from sim.genesis import ParametricResonance, LeptogenesisModel, MatterGenesisSimulation
    
    # Parametric resonance
    pr = ParametricResonance(inflaton_mass=1e13, coupling=1e-7)
    rate = pr.particle_production_rate(phi_amplitude=1e16, k=1.0)
    print(f"Particle production rate: {rate:.2e}")
    
    # Leptogenesis
    lepto = LeptogenesisModel(M=1e10, Yukawa=1e-6, CP_violation=1e-6)
    result = lepto.solve_leptogenesis()
    print(f"Baryon asymmetry η_B: {result['eta_B']:.2e}")
    print(f"Observed value: {result['eta_observed']:.2e}")
    
    # Full simulation (quick version)
    sim = MatterGenesisSimulation()
    history = sim.evolve_universe(total_time=100, dt=1.0)
    summary = sim.get_summary(history)
    
    print(f"\nMatter genesis after {summary['n_steps']} steps:")
    print(f"  Final temperature: {summary['final_temperature']:.2e} GeV")
    for comp, frac in summary['composition'].items():
        print(f"  {comp}: {frac:.2%}")


def holographic_example():
    """Holographic analysis example."""
    print("\n" + "=" * 60)
    print("HOLOGRAPHIC ANALYSIS")
    print("=" * 60)
    
    from sim.holographic import HolographicAnalysis, UniverseFormulaReport
    from sim.constants import CosmologicalConstants
    
    # Basic analysis
    analysis = HolographicAnalysis()
    results = analysis.analyze_all_models()
    
    print("k-alpha analysis across cosmological models:")
    print(f"  Mean k: {results['mean_k']:.6f}")
    print(f"  k/α: {results['mean_k_over_alpha']:.2f}")
    print(f"  Error vs 66α: {results['mean_error_vs_66alpha']:.2f}%")
    
    # Formula comparison
    comparison = analysis.formula_comparison()
    print(f"\nBest k formula: {comparison['best_formula']}")
    
    # Constants summary
    c = CosmologicalConstants()
    print(f"\nKey relations:")
    print(f"  α = {c.alpha:.10f}")
    print(f"  1/α = {c.inverse_alpha:.4f}")
    print(f"  k_observed = {c.k_observed:.6f}")
    print(f"  k/α = {c.k_over_alpha:.2f}")


def main():
    """Run all examples."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     Unified Cosmological Simulation - Quick Start Guide      ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    quantum_example()
    cosmic_example()
    coherence_example()
    genesis_example()
    holographic_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

