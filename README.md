# Classification of Gravitational Waves

This project focuses on the **classification of gravitational wave signals**, specifically targeting **Binary Black Hole (BBH)**, **Binary Neutron Star (BNS)**, and **Neutron Star-Black Hole (NSBH)** events. By combining **synthetic data generation**, **noise injection**, and **deep learning techniques**, this project provides a robust pipeline for classifying gravitational wave signals with high accuracy.

---

## 🚀 Project Overview

Gravitational waves, ripples in spacetime caused by massive celestial events, are vital for understanding the universe. This project aims to:
- Generate **synthetic gravitational wave signals** using advanced simulation tools.
- Inject realistic **Gaussian noise** to mimic real-world conditions.
- Train a **Long Short-Term Memory (LSTM)** model to classify gravitational wave events.
- Test the model's performance on both **synthetic** and **real data** (e.g., from [GWOSC](https://www.gw-openscience.org)).

---

## 📂 Project Structure

```plaintext
├── data/
│   ├── Synthetic_Data_Generation_and_Preprocessing/
│   │   ├── Generation_of_Synthetic_Gravitational_Wave_Signals/  # Synthetic signals
│   │   ├── Noise/                                               # Noise injection processes
│   │   ├── Real_Data/                                           # Real gravitational wave data
│   │   └── Data_Analysis_and_Visualization.ipynb                # Notebook for data analysis and visualization
│
├── model/
│   ├── Model_Preparation_and_Training/
│   │   └── train_model_w_noise.py                               # Script for training the model
│
├── result/
│   ├── Model_Testing_and_Evaluation/
│   │   ├── testing_the_model_on_Generated_Data/                 # Testing on synthetic data
│   │   └── testing_the_model_on_Real-World_Data/                # Testing on real-world data
│
└── README.md                                                    # Project description and guide

 ```

---

## ⚙️ **Technologies and Tools**

Python Libraries: LALSimulation, PyCBC, GWpy, TensorFlow, NumPy, Matplotlib

Data Sources:
Synthetic data generated using LALSimulation

Real data from the Gravitational Wave Open Science Center (GWOSC)

Deep Learning Framework: TensorFlow/Keras (LSTM)
