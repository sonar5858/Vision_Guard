# Vision Guard 🛡️  
AI-powered cap detection system for automated quality control in industrial environments.

---

## 🔍 Project Overview

**Vision Guard** leverages YOLOv8 and Roboflow's local inference server to detect the presence or absence of caps in real-time. It’s designed for integration with industrial capping and bottling machinery to enhance quality control and reduce human error.
But YOLOv8 model does not worked or is under devlopement Though i used Roboflow instant model to train and seems to work 
---

## 🧠 Features

- 🧠 Cap and missing cap detection (`cap`, `missing_cap`)
- 🎯 Trained using a custom dataset on Roboflow
- 🐳 Local model inference using Docker
- 🎥 Real-time detection via webcam using `matplotlib`
- 🔐 Environment-based API key management (`.env`)
- 📈 Easily extendable to MySQL logging and MQTT alerts

---
