# btfsim

Beam dynamics simulations in the Spallation Neutron Source (SNS) Beam Test Facility (BTF) using [pyorbit](https://github.com/PyORBIT-Collaboration/py-orbit). 

Mostly copied/adapted from K. Ruisard's repo. 

* **./btfsim**: lattice/bunch generation, tracking, etc.
* **./data**: small data files like lattice XML.
* **./scripts** various simulation scripts
    * Generic scripts (e.g. load/track bunch) are contained in undated files (e.g. 'track.py')
    * Specific studies/benchmarks are contained in dated folders (e.g. '/scripts/YYYY-MM-DD/'), optionally with a descriptive tag ('/scripts/YYYY-MM-DD_description/')
        * Scripts in these folders do not need to be dated.
        * Each folder should have a README describing the study.
        * Analysis notebooks should be kept here as well. Jupyter notebooks should be cleared before commiting changes.

No input/output data files are tracked with git; data will be stored on DropBox or external drive with the same file structure. Output files from bunch tracking should have format '{YYMMDDHHMMSS}-{script_name}-{start_node}-{stop_node}-{data_type (like 'bunch' or 'history')}-{location_in_lattice}.dat'.

Data analysis/visualization routines will be kept in the [`beamphys`](https://github.com/austin-hoover/beamphys) repository. 