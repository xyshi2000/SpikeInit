# Training Deep Spiking Neural Networks without Normalization

## Installing Dependencies

```bash
pip3 install torch torchvision
pip3 install tensorboard thop spikingjelly==0.0.0.0.14 cupy-cuda11x timm scipy
```

## Usage

The default configuration trains a VGG-11 on CIFAR10 with SpikeInit. To start training, simply run:

```bash
python main.py
```

You can also specify the model, the initialization method, the normalization, and other hyperparameters for comparison.

```bash
python main.py --model <model_name> --init-method <initialization_method> --conv <convolutional_layer> --activation <spiking_neuron>
```

For example, to train MS-ResNet-18 on CIFAR10 with tdBN, run:

```bash
python main.py --model ms_resnet18_tiny --init-method kaiming --conv ConvBN --activation LIF --zero-init-residual false
```

Detailed configuration options are listed in the following table:

| Method                     | init-method        | conv     | activation | zero-init-residual |
| -------------------------- | ------------------ | -------- | ---------- | ------------------ |
| Fluctuation-driven init    | fluctuation_driven | Conv     | LIF        | true               |
| Ding et al. (2025) init    | ding               | Conv     | LIF        | true               |
| Micheli et al. (2025) init | micheli            | Conv     | LIF        | true               |
| BNTT                       | kaiming            | ConvBNTT | LIF        | false              |
| tdBN                       | kaiming            | ConvBN   | LIF        | false              |
| TEBN                       | kaiming            | ConvTEBN | LIF        | false              |
| MPBN                       | kaiming            | ConvBN   | MPBNLIF    | false              |
| SpikeInit                  | spiking            | Conv     | ASLIF      | true               |

## Implementation Details

The core implementation of **SpikeInit** is located in `models/submodules/initialization.py`.

The function `_steady_state_equations` implements the system of nonlinear equations from **Theorem 4.1** in the paper. These equations are solved by `_find_steady_state`, following **Algorithm 1**.

The weight initialization parameter $\sigma^*$ is computed in `calculate_parameters`. This function first performs a binary search for $\sigma^*$ using `_find_sigma`, following **Algorithm 2**. It then computes $\sigma_w^{(1)}$ for the direct coding layer using `_find_static_sigma`.

The initial shape parameter of the surrogate gradient is computed in `calculate_alpha`. This function provides a general numerical approach for estimating the expected squared derivative of the surrogate function, denoted by $\mathcal{M}_2$, via numerical integration. To support other surrogate functions, you can modify the corresponding `integrand`. The function then solves the nonlinear equation from **Theorem 5.1** to obtain $\alpha^*$.

The simulation-based weight initialization is implemented in `calculate_parameters_sim`. It generates random input currents and simulates the membrane potential dynamics using `_simulate`. The function `_find_sigma_sim` then performs a binary search to identify $\sigma^*$, following **Algorithm 3**.

The simulation-based surrogate-gradient initialization is implemented in `calculate_alpha_sim`. It estimates $\mathcal{M}_2$ using `_compute_M2_numerical`, based on membrane potential samples collected from the simulations.

## Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@inproceedings{shi2026training,
  title={Training Deep Spiking Neural Networks without Normalization},
  author={Shi, Xinyu and Yu, Zhaofei},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
