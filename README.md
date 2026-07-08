# Autonomous Passenger Safety System

An AI-powered passenger monitoring system designed to enhance the safety of autonomous vehicles using Computer Vision and Deep Learning. The system monitors passenger behaviour and health conditions in real time by integrating YOLOv8 object detection, pose estimation, eye blink detection, and ECG-based heart rate monitoring.

## Project Overview

This project aims to improve passenger safety inside autonomous vehicles by continuously monitoring passenger activities and physiological conditions. It detects abnormal behaviour such as drowsiness, unconsciousness, and medical emergencies, enabling timely alerts and emergency response.

## Features

- Real-time passenger monitoring
- YOLOv8 object detection
- YOLOv8 pose estimation
- Eye blink detection
- ECG-based heart rate monitoring
- Emergency alert generation
- Real-time behaviour analysis
- AI-assisted safety monitoring

## Technologies Used

- Python
- YOLOv8
- OpenCV
- PyTorch
- NumPy
- Ultralytics
- ZED2 Camera
- ECG Sensor

## Project Structure

```
Autonomous-Passenger-Safety-System
│
├── code/
├── dataset/
├── docs/
├── project-designs/
├── reference-papers/
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

1. Clone the repository.

```bash
git clone https://github.com/Gvikram25/Autonomous-Passenger-Safety-System.git
```

2. Navigate to the project directory.

```bash
cd Autonomous-Passenger-Safety-System
```

3. Install the required dependencies.

```bash
pip install -r requirements.txt
```

## Usage

Run the required Python scripts from the `code` folder after installing all dependencies and connecting the required hardware components.

```bash
python sender.py
```

or

```bash
python receiver.py
```

## Dataset

The project was developed and tested using a custom dataset for passenger behaviour analysis. Due to GitHub storage limitations, the complete dataset is not included in this repository.

## Results

The system successfully detects passenger behaviour and monitors health-related conditions in real time using computer vision and ECG sensor data. It provides timely alerts to enhance passenger safety inside autonomous vehicles.

## Future Enhancements

- Cloud-based monitoring
- Mobile application integration
- GPS-based emergency assistance
- Driver and passenger emotion detection
- Improved deep learning models for higher accuracy

## Documentation

Project-related documents are available in the `docs` folder.

- Project Report
- Project Poster
- Setup Guide

System design documents are available in the `project-designs` folder.

Research papers referred to during development are available in the `reference-papers` folder.

## Author

**Gs Vikram**

GitHub: https://github.com/Gvikram25

## License

This project is developed for academic and educational purposes as part of a final-year capstone project.
