
<br />
<div align="center">
  <h2 align="center">Receptive Field Mapping for VISp (Calcium Imaging)</h2>
  <p align="center">
    A Python tool to map receptive fields from Allen Brain Observatory data using locally sparse noise stimuli.
    <br />
    <strong>Preliminary tool for upcoming analysis of proprietary calcium imaging data</strong>
    <br />
    <br />
    <a href="https://portal.brain-map.org/explore/circuits/visual-coding-2p">Allen Brain Observatory</a>
  </p>
</div>

---

## About The Project

This script provides a working pipeline to estimate spatial receptive fields from two-photon calcium imaging data collected in the primary visual cortex (VISp) of mice. The current version uses data from the Allen Brain Observatory as a placeholder while awaiting delivery of a custom dataset. RF maps are computed based on responses to locally sparse noise stimuli.

This project was developed as part of a research collaboration under the supervision of Dr. Madineh Sedigh-Sarvestani.

### Built With

* [Python 3](https://www.python.org/)
* [AllenSDK](https://allensdk.readthedocs.io/en/latest/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)

---

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.9**  
  This project is compatible with Python 3.9. You can check your version with:

  ```bash
  python3 --version
  ```

- **[Homebrew](https://brew.sh/)** (macOS only)  
  Used for installing system dependencies such as HDF5. If you don't have it installed:

  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

- **HDF5**  
  Required by certain `allensdk` dependencies (e.g., `tables`). Install via Homebrew:

  ```bash
  brew install hdf5
  ```

- **Virtual Environment**  
  It is recommended to use a virtual environment to avoid dependency conflicts:

  ```bash
  python3.9 -m venv allensdk-env
  source allensdk-env/bin/activate
  ```

- **AllenSDK**  
  Once inside the virtual environment and with HDF5 installed, run:

  ```bash
  HDF5_DIR=/opt/homebrew/opt/hdf5 pip install allensdk
  ```

- **(Optional) PostgreSQL**  
  If you encounter errors related to `pg_config` while installing `psycopg2`, install PostgreSQL:

  ```bash
  brew install postgresql
  ```

> If you're using PyCharm, make sure to configure your interpreter to point to the `allensdk-env` environment you created above.

### Installation

1. Clone this repository (or copy the script locally)
2. Make sure you have access to the Allen Brain Observatory cache
3. Set the correct path to your `manifest.json` file in the script

### Usage

```bash
python rf_mapping_script.py
```

The script will:
- Load VISp experiment using locally sparse noise stimulus
- Compute cell-wise receptive field maps
- Display up to 16 example RF maps

---

## Contributing

Contributions are welcome, especially if you’re working with calcium imaging datasets or extending the analysis!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

Distributed under the Allen Institute's data license for public datasets. Custom code is open for academic use.

---

## Contact

Emily Ma – jm2786@cornell.edu    
Supervised by Dr. Madineh Sedigh-Sarvestani  
Based on AllenSDK by the Allen Institute for Brain Science
