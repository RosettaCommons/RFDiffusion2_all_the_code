# RFdiffusion2 for Molecular Interfaces

Open source code for RFdiffusion2 for Molecular Interfaces, an extension of [RFD2](https://www.nature.com/articles/s41592-025-02975-x), as described in the [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.09.29.678898v2).

# Quick start

You need Git, `curl`, an x86-64 Linux system with a supported NVIDIA GPU and CUDA 12.8-compatible driver, and [Apptainer](https://apptainer.org/docs/admin/main/installation.html).

1. Clone the repository and its submodules:

   ```bash
   git clone https://github.com/RosettaCommons/RFDiffusion2_all_the_code.git
   cd RFDiffusion2_all_the_code
   git submodule update --init --recursive
   export REPO_DIR="$PWD"
   ```

2. Download and verify the prebuilt container:

   ```bash
   ./rf_diffusion/exec/download.sh apptainer
   ```

   The SIF SHA-256 is `f8bdfd4e9570fe4091931512a2570b71729a110efdb7b908c7f2c67cfbb9b025`.

3. Download the published model weights:

   ```bash
   ./rf_diffusion/exec/download.sh weights
   export RFDIFFUSION2_WEIGHTS_DIR="$REPO_DIR/rf_diffusion/model_weights"
   ```

   This downloads `RFD_173.pt`, `RFD_140.pt`, the `RFD_45.pt` regression-test checkpoint, and the RFdiffusion2-MI checkpoint `ppi_robust_struct.pt`. Existing files are skipped; use `--force` to replace them. If you use `--output-dir DIR`, set `RFDIFFUSION2_WEIGHTS_DIR` to the absolute path of that directory.

   To download both the container and weights in one command, run `./rf_diffusion/exec/download.sh all`.

4. Run the portable GPU smoke test. It uses only the public checkout, bundled test data, and downloaded weights:

   ```bash
   (
       cd "$REPO_DIR/rf_diffusion"
       apptainer exec --nv exec/rf_diffusion_aa.sif pytest -q test_portable_smoke.py
   )
   ```

## Build the container locally instead

To build the image from `rf_diffusion/exec/rf_diffusion_aa.spec` rather than downloading it:

```bash
./rf_diffusion/exec/build_rf_diffusion_aa_apptainer.sh
```

The default build contains the runtime dependencies but not this repository or the model weights. Run with `--with-repo --with-weights` for a self-contained image, or `--help` for all options.

## Running repository scripts

Executable scripts such as `rf_diffusion/run_inference.py` automatically launch the container, set `PYTHONPATH`, and default `RFDIFFUSION2_WEIGHTS_DIR` to `rf_diffusion/model_weights`. If the SIF is missing, the wrapper offers to download it. For scripts without the container shebang, run:

```bash
apptainer exec --nv \
    --bind "$REPO_DIR:$REPO_DIR" \
    --env PYTHONPATH="$REPO_DIR" \
    --env RFDIFFUSION2_WEIGHTS_DIR="$RFDIFFUSION2_WEIGHTS_DIR" \
    rf_diffusion/exec/rf_diffusion_aa.sif python path/to/script.py ...
```

Set `RFDIFFUSION2_SIF_PATH` to use another SIF or `RFDIFFUSION2_APPTAINER_URI` to pull from another OCI URI.

# Protein binder design

This example designs a 65–120-residue protein binder against the human neonatal Fc receptor (FcRn). It uses the 151-residue, AlphaFold2-derived target from the paper's FcRn campaign and the same FcRn hotspots: B44, B49, B50, B56, and B58. The config also retains the campaign's helix sprinkling, late self-conditioning, custom timestep schedule, and extra timestep outputs.

```bash
./rf_diffusion/run_inference.py --config-name=fcrn_binder
```

The default one-design run writes its PDB and TRB under `outputs/fcrn_binder/`, together with pX0 snapshots at timesteps 30, 20, and 10. The tracked target is `rf_diffusion/examples/inputs/fcrn_human_af_B.pdb`.

# Cysteine protease design

This example scaffolds the ULP1-like catalytic residues His A113, Asp A130, and Cys A179 while designing a 180–200-residue protein against target residues B96–99. B96 is the interface hotspot. The config preserves the original custom timestep schedule and also writes pX0 snapshots at timesteps 30, 20, and 10.

From the repository root, run:

```bash
./rf_diffusion/run_inference.py --config-name=cysteine_protease
```

The default one-design run writes `outputs/cysteine_protease/design_0-atomized-bb-False.pdb`, the corresponding `.trb`, and the extra `_t{30,20,10}.pdb` snapshots. The input motif and target structure are tracked at `rf_diffusion/examples/inputs/ulp1_moitf_5.pdb`.

This example accompanies [De novo design of cysteine proteases](https://www.biorxiv.org/content/10.1101/2025.11.21.689808v2).

# Preparing the phosphotyrosine inputs

To regenerate the CD3epsilon and STAT5 peptide structures, run:

```bash
./rf_diffusion/examples/prepare_ptr_inputs.py
```

This trimmed PyRosetta script builds only the two annotated phosphotyrosine sequences, removes hydrogens, assigns chain B, and checks that PTR retains both native peptide bonds. Use `--output-dir DIR` to write copies somewhere other than `rf_diffusion/examples/inputs/`.

# CD3epsilon phosphotyrosine binder design

This example designs a 160-residue binder against the CD3epsilon peptide `PVPNPD-pY-EPIRKG`. PTR B7 is a flexible phosphotyrosine ligand, and its native peptide bonds to Asp B6 and Glu B8 are encoded by `CONECT` records in the tracked PDB. The two flanking residues are atomized and used as PPI hotspots because flexible ligand atoms cannot be selected with `ppi.hotspot_res`.

```bash
./rf_diffusion/run_inference.py --config-name=cd3epsilon_ptr
```

The default run writes `outputs/cd3epsilon_ptr/design_0-atomized-bb-False.pdb` and the corresponding `.trb`. The PDB contains the 160-residue designed chain, the `PVPNPD`-containing peptide residues, and PTR; the TRB also stores the atomized inference state with the two covalent peptide–PTR bonds.

# STAT5-pY694 binder design

This example designs a 160-residue binder against the STAT5 pY694 peptide `TPVLAKAVDG-pY-VKPQIKQVVP`. PTR B11 and its peptide bonds to Gly B10 and Val B12 are represented in the same way as the CD3epsilon example, with the two flanking residues used as PPI hotspots.

```bash
./rf_diffusion/run_inference.py --config-name=stat5_ptr
```

The default run writes `outputs/stat5_ptr/design_0-atomized-bb-False.pdb` and the corresponding `.trb`, containing the designed chain and retained PTR-containing peptide. The TRB stores the atomized inference state and covalent graph.

All four example configs default to `inference.num_designs=1` and use the downloaded `ppi_robust_struct.pt` checkpoint. To run a larger campaign, override the count and, if desired, the output prefix; for example:

```bash
./rf_diffusion/run_inference.py --config-name=cd3epsilon_ptr \
    inference.num_designs=100 \
    inference.output_prefix="$REPO_DIR/outputs/cd3epsilon_ptr_campaign/design"
```

# Citations

Please cite the RFdiffusion2 for Molecular Interfaces preprint:

```bibtex
@article{Bauer2025.09.29.678898,
  author={Bauer, Magnus S and Zhang, Jason Z and Wu, Kejia and Lee, Gyu Rie and Coventry, Brian and Silvestri, Isabella M and Klupt, Kody A and Shi, Jiuhan and Brent, Rafael I and Li, Xinting and Moller, Carolina and Roullier, Nicole and Vafeados, Dionne K and Kalvet, Indrek and Skotheim, Rebecca K and Zhu, Siyu and Motmaen, Amir and Herrmann, Luca C and Sturmfels, Pascal and Tischer, Doug and Altae-Tran, Han and Juergens, David and Krishna, Rohith and Ahern, Woody and Yim, Jason and Bera, Asim K and Kang, Alex and Joyce, Emily and Lu, Andrew and Stewart, Lance and DiMaio, Frank and Mudumbi, Krishna C and Baker, David},
  title={De novo design of phosphotyrosine peptide binders},
  elocation-id={2025.09.29.678898},
  year={2026},
  doi={10.1101/2025.09.29.678898},
  publisher={Cold Spring Harbor Laboratory}
}
```

This code extends RFdiffusion2; please also cite its preprint:

```bibtex
@article{ahern2025atom,
  title={Atom level enzyme active site scaffolding using RFdiffusion2},
  author={Ahern, Woody and Yim, Jason and Tischer, Doug and Salike, Saman and Woodbury, Seth M and Kim, Donghyo and Kalvet, Indrek and Kipnis, Yakov and Coventry, Brian and Altae-Tran, Han Raut and others},
  journal={bioRxiv},
  pages={2025--04},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
