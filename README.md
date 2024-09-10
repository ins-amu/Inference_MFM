This is a repo to make inference on a mean-filed model of spiking QIF neurons, using Optimization (DE/PSO), Simulation-based Inference (SBI), Hamiltonian Monte Carlo (HMC), and Neural ODEs.


For fast simulation (JIT/Jax), see the folder <em> Simulation</em>.

For Linear stability analysis, see the folder <em> LSA</em>.


Based on our benchmark, we recommend using SBI for efficient inference on generative parameters. Please refer to the folder <em>SBI</em> for a demonstration.


For exact (but computionally expensive) inference, see the folder <em> HMC</em>.


For fast point estimation, see the folder <em> Optimization</em>.



To infer the dynamics from simulations, particularly to learn vector fields and nullclines (travesering across scales), see the folder <em> Neural ODEs</em>.


This research has received funding from EU’s Horizon 2020 Framework Programme for
Research and Innovation under the Specific Grant Agreements No. 101147319 (EBRAINS
2.0 Project), No. 101137289 (Virtual Brain Twin Project), and government grant managed
by the Agence Nationale de la Recherch reference ANR-22-PESN-0012 (France 2030 program).

