# 🌱 Crop Disease RL Environment

## 📌 Problem
Build an OpenEnv RL environment for crop disease detection using images.

## 🧠 Approach
- CNN (ResNet18) for classification
- RL reward system for correctness

## 🎯 Reward System
- +1 → correct prediction
- -1 → incorrect prediction

## 📊 Dataset
PlantVillage dataset from Kaggle

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python train.py
python inference.py