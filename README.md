This is a repo to make inference on a mean-filed model of spiking QIF neurons, using Optimization (DE/PSO), Simulation-based Inference (SBI), Hamiltonian Monte Carlo (HMC), and Neural ODEs.


For fast simulation (JIT/Jax), see the folder <em> Simulation</em>.

For Linear stability analysis, see the folder <em> LSA</em>.


Based on our benchmark, we recommend using SBI for efficient inference on generative parameters. Please refer to the folder <em>SBI</em> for a demonstration.


For exact (but computionally expensive) inference, see the folder <em> HMC</em>.


For fast point estimation, see the folder <em> Optimization</em>.



To infer the dynamics from simulations, particularly to learn vector fields and nullclines (travesering across scales), see the folder <em> Neural ODEs</em>.
