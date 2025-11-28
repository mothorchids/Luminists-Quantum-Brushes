# Luminists-Quantum-Brushes
Supplementary Resources for Bradford Quantum Hackathon Final Submission. Refer to this [repository](https://github.com/mothorchids/QuantumBrush) for the implementation
and this [PR](https://github.com/moth-quantum/QuantumBrush/pull/25) for the integration of our two brushes.

## Usage

### Steerable
Click and drag on the canvas to select the copy region.
Click and drag to select the position of the targer region.
Either click or click and drag to select a paste region.

The effect is controlled by four parameters:
- t → time parameter for the evolution of the system
    - t = 0 → fully similar to the source
    - t = 1 → similar to the target
    - t > 1 → evolves beyond the target state, which may produce interesting effects
- timestep → the number of discrete time steps used to approximate the circuit’s evolution
- controls → the number of features used to represent a patch, corresponding to the number of qubits
- `Source = Paste` → whether to make the source region equal to the paste region

The user also has optional visualization controls: enable show `source & target` and optionally `show color` to display the reference regions.

Below is an application of Steerable brush to Joan Miró’s _El Jardín_. 
|![miro_el_garden](https://github.com/user-attachments/assets/64b4d1b7-3acc-4128-b178-fd78f5c57fb6)|<img width="736" height="872" alt="el_garden_miro_steerable" src="https://github.com/user-attachments/assets/32f8ade1-4ae5-4af3-b6fa-96b2b5588944" />|
|----------------|----------------|

### Chemical

Click and drag on the canvas to draw one or more strokes.

The effect is controlled by three parameters:
- Radius → determines the size of the brush
- Number of repetitions → determines the degree of smoothness of the effect
- Bond Distance → determines the interatomic distance to simulate

Below is an application of Chemical to Anita Malfatti's _Portrait of Mário de Andrade_.

<img width="700" height="955" alt="anita50_big_radius" src="https://github.com/user-attachments/assets/e45d3f4e-3060-43ac-83dd-08ba06d61107" />

## TODO
- Integration with quantum hardware and emulators.
  
