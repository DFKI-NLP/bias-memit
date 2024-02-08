# ROME
This package is an implementation of the Rank-One Model Editing (ROME) knowledge update (Meng et al. (2022), the predecessor work to MEMIT.  

All scripts in this directory are written by Meng et al. (2022).
The complete source code can be found in the follwing repository: [`https://github.com/kmeng01/rome`](https://github.com/kmeng01/rome).  

Although the ROME update is not performed in this thesis, the MEMIT code makes use of some of the functionalities and therefore the package is included in this repository.

The scripts in this package provide similar functionalities for ROME which the [`memit`](../memit) package does for MEMIT.
* [`compute_u.py`](compute_u.py): Allows to select vector $u$.
* [`compute_v.py`](compute_v.py): Finds an optimised $v_*$ and computes $v$.
* [`rome_main.py`](rome_main.py): Main update.
* [`rome_params.py`](rome_hparams.py): Interface for ROME hyperparameters. Inherits from the base [`params.py`](../util/hparams.py) module.

For the estimation of second moment statistics of keys ($C = KK$), Meng et al. (2022) provide the `layer_stats` module (see the [main README](../README.md)).

__*Source*__:
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. "Locating and Editing Factual Associations in GPT." Advances in Neural Information Processing Systems 36 (2022).

