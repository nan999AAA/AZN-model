# Train the initial AZN model

这是 AZN 模型训练的代码，相关代码还包括 `20260821 aiweatherquest-weeklysubmit-example` 和模型推理和四分位数计算。

## 目录

1. [简介](#1-简介)
2. [文件夹包含文件的说明](#2-文件夹包含文件的说明)
3. [运行说明](#3-运行说明)
4. [注意事项](#4-注意事项)
5. [参考文献](#5-参考文献)
6. [引用方式](#6-引用方式)

---

## 1. 简介

本文件包括训练代码和训练代码所需的输入数据。运行训练代码后，可以得到模型推理所需的模型。

训练得到的模型结构如下：

<p align="center">
  <img src="model%20name.png" alt="AZN Model">
</p>

*图：AZN 模型结构。模型图来源于20260724讨论.pdf第 8 页。*

---

## 2. 文件夹包含文件的说明

本文件夹按照预报变量和模型类型进行组织。

- `T2M/`：2 m temperature（T2M）相关模型的训练代码和数据。
  - `simpleCNN/`：Simple CNN 模型
  - `CG/`：ConvGRU 模型
  - `I/`：I 模型
  - `tc/`：Transformer-CNN 模型
  - `U-net/`：U-Net 模型
  - `SA1/`：SA1 模型
  - `D.nc`：T2M 模型训练数据
  - `D.ipynb`：T2M 数据处理相关 Notebook

- `MSLP/`：mean sea level pressure（MSLP）相关模型的训练代码和数据。
  - `1 u/`：模型 1
  - `2 SA/`：模型 2
  - `3 TC/`：模型 3
  - `4 upCG3/`：模型 4
  - `mslp_2D.nc`：MSLP 模型训练数据

- `PR/`：precipitation（PR）相关模型的训练代码和数据。
  - `U-NET/`：U-Net 模型
  - `SA/`：SA 模型
  - `tp_2D.nc`：PR 模型训练数据

- `README.md`：总体说明文件。
- `requirements.txt`：Python 环境及依赖配置文件。
- `environment.yml`：Conda 环境配置文件，与 `requirements.txt` 二选一。

---

## 3. 运行说明

### **3.1 环境配置**

使用 Conda：

    conda env create -f environment.yml
    conda activate <环境名称>

或使用 pip：

    pip install -r requirements.txt

### **3.2 模型运行**

从项目根目录运行对应模型的 Python 脚本。

T2M：

    python T2M/simpleCNN/T2M-simpleCNN-subseasonal_model_cpuc.py

其他 T2M 模型：

    python T2M/<模型目录>/<脚本名称>.py

MSLP：

    python "MSLP/<模型目录>/<脚本名称>.py"

PR：

    python PR/<模型目录>/<脚本名称>.py

训练数据：

    T2M/D.nc
    MSLP/mslp_2D.nc
    PR/tp_2D.nc

> 具体脚本名称以各模型文件夹中的实际 Python 文件为准。

---

## 4. 注意事项

（1）模型的输出格式和参数不一样，但都在代码对应的文件内。

（2）有些模型代码的训练过程较为原始，需运行完之后才能得到一个模型。

（3）代码中计算资源的配置默认禁止 GPU，暂无其他设置。

---

## 5. 参考文献

【在此填写参考文献】

---

## 6. 引用方式

【在此填写引用方式】
