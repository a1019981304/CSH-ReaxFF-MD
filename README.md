# CSH-ReaxFF-MD

## Overview
This repository hosts the open-source post-processing code for **ReaxFF reactive molecular dynamics (MD) simulations of Calcium Silicate Hydrate (C-S-H, SiO2) interfacial systems**, with a core focus on identifying interatomic chemical bonding configurations, quantifying hydrogen bond networks, and analyzing nanoscale interfacial structural properties.

Tailored for cementitious material research, this code enables fully automated processing of raw LAMMPS ReaxFF simulation outputs, systematic classification of atoms and chemical bonds, and high-fidelity statistical analysis of key structural descriptors. Raw intermediate data files from core workflows are also provided for reproducibility and validation.

## Key Features
### 1. Automated Atom & Bond Classification (`h2o_bond.py`)
- Parses raw LAMMPS trajectory files (`.lammpstrj`) and ReaxFF bond order outputs
- Classifies atoms into 19 distinct, system-specific types for CSH-silica interface systems (including structural/surface oxygen, interfacial water, hydroxyl groups, and calcium ions)
- Identifies covalent bonds (Si-O, Ca-O, O-H) via dual criteria: bond order (BO) thresholds with geometric distance fallback for robustness
- Generates standardized, frame-wise LAMMPS data files with classified atom types and validated bond information for downstream analysis

### 2. Comprehensive Structural & Interfacial Analysis (`bond_RDF_密度.py`)
- **Radial Distribution Function (RDF)**: Computes and block-averages RDF for 40+ targeted atomic pairs, with built-in Savitzky-Golay smoothing (with moving average fallback for compatibility) to reduce numerical noise
- **Coordination Number (CN)**: Integrates RDF curves to derive CN, with automated first/second valley detection for objective cutoff radius determination
- **Hydrogen Bond Quantification**: Implements standard geometric criteria (H···O distance + O–H···O angle) for hydrogen bond identification, supporting one-per-donor counting and area/volume normalization for interfacial systems
- **Interfacial Density Profiling**: Calculates 1D number and mass density profiles along the interface normal, with block averaging to quantify statistical uncertainty
- **Bond Dynamics Tracking**: Monitors the temporal evolution of covalent bonds (Si-O, Ca-O, O-H) across the full simulation trajectory

### 3. Statistical Rigor & Cross-Platform Compatibility
- Implements block averaging for all derived properties to quantify standard error (SE) for academic publication
- Fallback implementations for core numerical functions (smoothing, integration) to ensure full functionality without optional SciPy dependencies
- Exports all results in CSV and Excel formats with UTF-8 encoding for direct plotting and further analysis
- Fully compatible with standard LAMMPS ReaxFF simulation outputs

## Repository Structure
| File Name | Description |
|-----------|-------------|
| `h2o_bond.py` | Core script for atom classification, bond identification, and frame-wise LAMMPS data file generation from raw simulation outputs |
| `bond_RDF_密度.py` | Post-processing script for RDF, coordination number, hydrogen bond, and interfacial density profile analysis |
| `bonds_counts_per_frame.csv` | Example intermediate output: per-frame covalent bond counts across the simulation trajectory |
| `bonds_counts_block_mean_se.csv` | Example intermediate output: block-averaged covalent bond counts with corresponding standard error |
| `hbonds_counts_per_frame.csv` | Example intermediate output: per-frame hydrogen bond counts across the simulation trajectory |
| `hbonds_counts_block_mean_se.csv` | Example intermediate output: block-averaged hydrogen bond counts with corresponding standard error |
| `density_z_mass_combined_excelfriendly.csv` | Example intermediate output: combined mass density profile along the interface normal (z-axis), with Excel-compatible formatting |
| `LICENSE` | Full Apache 2.0 License for the repository |

## Environment & Requirements
### Core Requirements
- Python 3.8 or higher
- `numpy >= 1.21.0`

### Optional Dependencies (Enhanced Functionality)
- `scipy >= 1.7.0`: For optimized Savitzky-Golay smoothing and numerical integration
- `pandas >= 1.4.0` + `openpyxl >= 3.0.0`: For Excel file export and batch data processing

### Input Data Prerequisites
- LAMMPS trajectory file (`.lammpstrj`) from ReaxFF MD simulations
- ReaxFF bond order output file containing per-atom bond information
- Simulation box with 3D periodic boundary conditions (PBC)

## Workflow & Usage
The post-processing pipeline consists of two sequential, modular steps.

### Step 1: Atom Classification & Bond Identification
Run `h2o_bond.py` to process raw simulation outputs and generate classified per-frame data files.
1. **Configure Global Parameters**
   Edit the `Global Configuration Parameters` section in `h2o_bond.py` to set:
   - File paths for your LAMMPS trajectory (`TRAJ_FILE`) and ReaxFF bond file (`BOND_FILE`)
   - Working directory (`WORK_PATH`) and output file prefix
   - Number of simulation frames to process (`N_FRAME`)
   - Bonding criteria for Si-O, O-H, and Ca-O pairs
   - Initial and final atom type mappings matching your simulation system

2. **Execute the Script**
   ```bash
   python h2o_bond.py
   ```
   This generates a classified LAMMPS data file (`data_frame_classified_{i}.data`) for each simulation frame, which serves as the input for the subsequent structural analysis.

### Step 2: Structural & Interfacial Property Analysis
Run `bond_RDF_密度.py` to compute and export all structural descriptors from the classified data files.
1. **Configure Analysis Parameters**
   Edit the global configuration section in `bond_RDF_密度.py` to set:
   - Working directory matching the output path from Step 1
   - Frame range and block size for statistical averaging
   - Hydrogen bond geometric criteria (distance and angle thresholds)
   - RDF parameters (bin width, maximum calculation radius)
   - Interface window settings for Interfacial Transition Zone (ITZ) analysis

2. **Execute the Script**
   ```bash
   python bond_RDF_密度.py
   ```
   All analysis results will be automatically exported to the `analysis_out` subdirectory in your working path.

## Output Files Description
All results are exported in CSV format (with Excel-compatible UTF-8 encoding). Key outputs include:
1. **Bond & Hydrogen Bond Statistics**
   - Frame-wise and block-averaged counts of covalent bonds and hydrogen bonds, with area/volume normalization for interfacial systems
2. **RDF & Coordination Number**
   - Per-pair RDF curves with mean values and standard error
   - Coordination number curves (raw and smoothed) for each atomic pair
   - Summary table of first/second valley positions and corresponding coordination numbers
3. **Density Profiles**
   - 1D number and mass density (g/cm³) profiles along the interface normal
4. **Interfacial Metrics**
   - Temporal evolution of interface position, window size, and atomic coverage across the simulation trajectory

## License
This project is licensed under the **Apache License 2.0** - see the LICENSE file for full license text and terms.

## Citation
If you use this code or intermediate data in your research, please cite this repository:
```
GitHub Repository: https://github.com/a1019981304/CSH-ReaxFF-MD
```
For corresponding research work citation, please refer to the original publication linked in the repository (to be updated).

## Contact
For questions, bug reports, or collaboration inquiries, please open an issue in this repository or contact the repository maintainer via GitHub.
