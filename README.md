# btf-simulation

Beam dynamics simulations in the Spallation Neutron Source (SNS) Beam Test Facility (BTF) using PyORBIT.

Adapted from K. Ruisard's repo.

* **./btfsim**: lattice/bunch generation, tracking, etc.
* **./data**: small data files like lattice XML.
* **./scripts** various simulation scripts
    * Generic scripts (e.g. load/track bunch) are contained in undated files (e.g. 'track.py')
    * Specific studies/benchmarks are contained in dated folders (e.g. '/scripts/YYYY-MM-DD/'), optionally with a descriptive tag ('/scripts/YYYY-MM-DD_description/')
        * Scripts are contained in dated files with descriptive tag (e.g. '2022-09-01_make_bunch.py')
        * Each folder should have a README.txt or README.md describing the study.
        * All analysis scripts/notebooks are kept here as well. Jupyter notebooks should be cleared before commiting changes.
        * No data is tracked with git; data will be stored on DropBox or external drive with same file structure as the `scripts` folder.
