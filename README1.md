# **Train the Initial AZN Model**

This folder contains the training code for the AZN model. The related code also includes `20260821 aiweatherquest-weeklysubmit-example`, as well as the code for model inference and quartile calculation.

## **Contents**

1. [Introduction](#1-introduction)
2. [Folder Contents](#2-folder-contents)
3. [Running Instructions](#3-running-instructions)
4. [Notes](#4-notes)
5. [References](#5-references)
6. [Citation](#6-citation)

---

## **1. Introduction**

This folder contains the training code and the input data required for model training. After running the training code, the models required for model inference can be obtained.

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
  * `D.nc`: T2M model training data
  * `D.ipynb`: Notebook for T2M data processing

* `MSLP/`: Training code and data for mean sea level pressure (MSLP) models.

  * `1 u/`: Model 1
  * `2 SA/`: Model 2
  * `3 TC/`: Model 3
  * `4 upCG3/`: Model 4
  * `mslp_2D.nc`: MSLP model training data

* `PR/`: Training code and data for precipitation (PR) models.

  * `U-NET/`: U-Net model
  * `SA/`: SA model
  * `tp_2D.nc`: PR model training data

* `Monthly/`: Monthly data processing code and data.

  * `T2M&MSLP MONTH.py`: Monthly T2M and MSLP data processing
  * `PR MONTH.py`: Monthly PR data processing
  * `merged_1.5deg_no_norm.nc`: Merged monthly data without normalization
  * `merged_1.5deg_no_norm_with_tp.nc`: Merged monthly data without normalization, including PR
  * `.DS_Store`: macOS system file

* `README.md`: General documentation

* `requirements.txt`: Python environment and dependency configuration

* `environment.yml`: Conda environment configuration; use either `environment.yml` or `requirements.txt`

---

## **3. Running Instructions**

### **3.1 Environment Setup**

Using Conda:

```
conda env create -f environment.yml
conda activate <environment_name>
```

Or using pip:

```
pip install -r requirements.txt
```

### **3.2 Model Running**

Run the corresponding Python script from the project root directory.

**T2M:**

```
python T2M/simpleCNN/T2M-simpleCNN-subseasonal_model_cpuc.py
```

**Other T2M models:**

```
python T2M/<model_directory>/<script_name>.py
```

**MSLP:**

```
python "MSLP/<model_directory>/<script_name>.py"
```

**PR:**

```
python PR/<model_directory>/<script_name>.py
```

**Monthly:**

```
python "Monthly/T2M&MSLP MONTH.py"
python "Monthly/PR MONTH.py"
```

**Training data:**

```
T2M/D.nc
MSLP/mslp_2D.nc
PR/tp_2D.nc
```

**Monthly data:**

```
Monthly/merged_1.5deg_no_norm.nc
Monthly/merged_1.5deg_no_norm_with_tp.nc
```

> The exact script names should be checked in the corresponding model folders.

---

## **4. Notes**

1. The output formats and parameters vary between models and are specified in the corresponding code files.

2. Some model implementations are relatively basic. A trained model is generated only after the training process is completed.

3. GPU usage is disabled by default in the code. No additional computing-resource configuration is currently provided.

---

## **5. References**

[References to be added]

---

## **6. Citation**

[Citation information to be added]

