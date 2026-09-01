# **Train the Initial AZN Model**

This folder contains the training code for the AZN model, together with the related code in `20260821 aiweatherquest-weeklysubmit-example` and the code for model inference and quartile calculation.

## **Contents**

1. [Introduction](#1-introduction)

2. [Folder Contents](#2-folder-contents)

3. [Running Instructions](#3-running-instructions)

4. [Notes](#4-notes)

5. [References](#5-references)

6. [Citation](#6-citation)

---

## **1. Introduction**

This folder contains the training code and the input data required for model training. After running the training code, the trained models required for model inference can be obtained.

The trained model architecture is shown below:

<p align="center">

  <img src="model%20name.png" alt="AZN Model">

</p>

*Figure: AZN model architecture. The model diagram is from page 8 of `20260724讨论.pdf`.*

---

## **2. Folder Contents**

This folder is organized according to forecast variables and model types.

* `T2M/`: Training code and data for 2 m temperature (T2M) models.

  * `simpleCNN/`: Simple CNN model
  * `CG/`: ConvGRU model
  * `I/`: I model
  * `tc/`: Transformer-CNN model
  * `U-net/`: U-Net model
  * `SA1/`: SA1 model
  * `D.nc`: T2M training data
  * `D.ipynb`: Notebook for T2M data processing

* `MSLP/`: Training code and data for mean sea level pressure (MSLP) models.

  * `1 u/`: Model 1
  * `2 SA/`: Model 2
  * `3 TC/`: Model 3
  * `4 upCG3/`: Model 4
  * `mslp_2D.nc`: MSLP training data

* `PR/`: Training code and data for precipitation (PR) models.

  * `U-NET/`: U-Net model
  * `SA/`: SA model
  * `tp_2D.nc`: PR training data

* `README.md`: General documentation

* `requirements.txt`: Python environment and dependency configuration

* `environment.yml`: Conda environment configuration; use either `environment.yml` or `requirements.txt`

---

## **3. Running Instructions**

### **3.1 Environment Setup**

Using Conda:

```bash
conda env create -f environment.yml
conda activate <environment_name>
```

Or using pip:

```bash
pip install -r requirements.txt
```

### **3.2 Running the Models**

Run the corresponding Python script from the project root directory.

**T2M:**

```bash
python T2M/simpleCNN/T2M-simpleCNN-subseasonal_model_cpuc.py
```

**Other T2M models:**

```bash
python T2M/<model_directory>/<script_name>.py
```

**MSLP:**

```bash
python "MSLP/<model_directory>/<script_name>.py"
```

**PR:**

```bash
python PR/<model_directory>/<script_name>.py
```

**Training data:**

```text
T2M/D.nc
MSLP/mslp_2D.nc
PR/tp_2D.nc
```

> The exact script names should be checked in the corresponding model folders.

---

## **4. Notes**

1. The output formats and model parameters vary between models and are specified in the corresponding code files.

2. Some model implementations are relatively basic, and the model files are generated only after the training process is completed.

3. The code is configured to disable GPU usage by default. No additional computing-resource settings are currently provided.

---

## **5. References**

[References to be added]

---

## **6. Citation**

[Citation information to be added]
