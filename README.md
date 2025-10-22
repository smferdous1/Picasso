# Picasso: Memory Efficient Parallel Graph Coloring for Quantum Chemistry

This repository provides sequential and GPU source code for the graph coloring algorithm introduced in the following publication:

```bibtex
@inproceedings{ferdous2024picasso,
  title={Picasso: Memory-Efficient Graph Coloring Using Palettes With Applications in Quantum Computing},
  author={Ferdous, S M and Neff, Reece and Peng, Bo and Shuvo, Salman and Minutoli, Marco and Mukherjee, Sayak and Kowalski, Karol and Becchi, Michela and Halappanavar, Mahantesh},
  booktitle={2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  pages={241--252},
  year={2024},
  organization={IEEE}
}
```

The graph coloring problem addressed here is motivated by quantum chemistry applications, specifically for efficient measurement through unitary partitioning of Pauli strings. Please cite the above publication if you use Picasso in your research.

## Installation

### Dependencies
Ensure you have the following dependencies installed:
- C++ compiler (e.g., g++) with C++17 support
  - g++ 9.1.0 is the earliest tested version known to work
- CMake (version >= 3.1)
- CUDA (version >= 11; tested, earlier versions may also work)

### Building Picasso
Clone and build the repository:

```bash
git clone git@github.com:smferdous1/Picasso.git --recurse-submodules
cd Picasso

# Create and enter build directory
mkdir build
cd build

# Run CMake configuration
cmake ..

# Compile the project
make
```

## Dataset

A sample dataset is included in JSON format under `data/pauli_ket_ccsd_data`. This dataset represents the hydrogen molecule (H) with varying numbers of bases and basis functions. The complete dataset used in our paper can be downloaded from [here](https://www.dropbox.com/scl/fi/h2d078dctva5tj845wbn3/Pauli_terms_H_clusters.zip?rlkey=n27whg3blu41wii28w06k5ahb&st=n8taqh18&dl=0).

## Running the Code

Two primary executables are provided:
- **CPU-only version**: `build/apps/palcolEr` (source: `apps/appPalColEncRecDir.cc`)
- **GPU-accelerated version**: `build/apps/palcolGr` (source: `apps/appPalColGpuMemRec.cc`)
- (Additional helper) `build/apps/palcolEcl` to test `.egr` on CPU directly.

Use the `-h` option with either executable to view available parameters and usage instructions.

### Examples

**Example 1:** GPU version help
```bash
./build/apps/palcolGr -h
```

**Example 2:** CPU-only execution

Generate grouping for the `Pauli_ket_ccsd_H2_631g.json` dataset using the recursive option (`-r`) with an initial target of 15 colors:

```bash
./build/apps/palcolEr -t 15 -r --in data/pauli_ket_ccsd_data/Pauli_ket_ccsd_H2_631g.json --out data/group_results/H2_631g_out_cpu.json
```

Make sure the `data/group_results` directory exists before running this command.

Expected screen output:
```
The grouping output: data/group_results/H2_631g_out1.json
using list greedy ordering for conflict coloring

***********Level 0*******
Num Nodes: 89
Num Edges: 1988
Avg. Deg.: 44.6742
Palette Size: 15
List Size: 4
Num Conflict Edges: 1539
Conflict to Edge (%): 77.4145
Num Colors: 15
Assign Time: 2.80626e-05
Conf. Build Time: 0.000160687
Conf. Color Time: 0.000172731

Final Num invalid Vert: 23
Naive Color TIme: 8.6166e-06
# of Final colors: 26
```

**Example 3:** GPU execution

Generate grouping for the `Pauli_ket_ccsd_H2_631g.json` dataset using GPU with
the recursive option (`-r`) with an initial target of 15 colors:

```bash
./build/apps/palcolGr -t 15 -r --in data/pauli_ket_ccsd_data/Pauli_ket_ccsd_H2_631g.json 
```

Expected screen output:
```
Using 32-bit offsets
Temp Storage Bytes: 767
Current Bytes: 360
Fits: 12312 < 31328
***********Level 0*******
Num Nodes: 89
List Size: 4
Num Conflict Edges: 1539
Num Colors: 15
Assign Time: 1.30097
Conf. Build Time: 0.017109
Conf. Color Time: 0.0001968

Final Num invalid Vert: 23
Naive Color TIme: 8.92766e-06
# of Final colors: 26
```

### ECLgraph (.egr) Support (CPU and GPU)

Picasso now accepts [ECL CSR graphs](https://userweb.cs.txstate.edu/~mb92/research/ECLgraph/index.html) via `include/ECLgraph.h`:
- CPU: `palcolEcl` reads `.egr` and runs the palette pipeline on CPU.
- GPU: `palcolGr` can also take `.egr`. It builds the conflict graph on GPU from the input CSR, then colors on CPU (timed separately).

Run on an ECL graph (sample files under `ecl_graph/`):
```bash
./build/apps/palcolGr --in ecl_graph/facebook.egr -t 15
```
You will see separate timings:
- Assign Time: list-color assignment (and device setup)
- Conf. Build Time: GPU build of the conflict graph from CSR (.egr)
- Conf. Color Time: CPU coloring on the conflict graph

Note: For `.egr` inputs, only conflict-graph construction is GPU-accelerated at present; coloring runs on CPU.

**Example 4:** Large GPU execution

Generate grouping for the `Pauli_ket_ccsd_H4_1D_631g.json` dataset using GPU with
the recursive option (`-r`), alpha=2 (`-a 2`), an initial target of 1% of the total number
of nodes (`-t 0.01`), and exporting the output grouping.

```bash
./build/apps/palcolGr -t 0.01 -a 2 -r --in data/pauli_ket_ccsd_data/Pauli_ket_ccsd_H4_1D_631g.json --out data/group_results/H2_631g_out_gpu.json
```

Expected screen output of the last level:
```
***********Level 67*******
Num Nodes: 106
List Size: 1
Num Conflict Edges: 2852
Num Colors: 3943
Assign Time: 1.38357e-05
Conf. Build Time: 0.00385322
Conf. Color Time: 3.38331e-05

Final Num invalid Vert: 99
Naive Color TIme: 0.00381275
# of Final colors: 4004

```

## Contact

For questions or further information, please contact:

- **S M Ferdous:** [sm.ferdous@pnnl.gov](mailto:sm.ferdous@pnnl.gov), [ferdous.csebuet@gmail.com](mailto:ferdous.csebuet@gmail.com)
- **Reece Neff:** [rwneff@ncsu.edu](mailto:rwneff@ncsu.edu), [reece.neff@pnnl.gov](mailto:reece.neff@pnnl.gov)
