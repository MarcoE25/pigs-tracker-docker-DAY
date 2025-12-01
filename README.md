# Pigs Tracker DIA 🐷

Pigs Tracking System and counting using YOLO + Kalman Filter + ReID. 
Includes logic for recovering identity after long occlusions and ID merging.

## Structure
- `models/`: models/: Contains the trained YOLO weights.
- `data/`: data/: Place your input video here.
- `main.py`: main.py: Main execution script.


## Execution
1.**FIRST DOWNLOAD THE CODE**: git clone https://github.com/MarcoE25/pigs-tracker-docker-DAY.git
(OPEN THE CODE (REPOSITORY) AND OPEN  A TERMINAL AND RUN THE NEXT STEP)
2.**BUILD THE CONTAINER**: docker build -t pigs-tracker .
3.**EXECUTE THE TRACKER USING GPU** REMBER TO CHANGE THE ROUTE: docker run --rm --gpus all -v "C:\ROUTE\TO\YOUR\DATA:/app/data" -v "C:\ROUTE\TO\YOUR\output:/app/output" pigs-tracker
**REMEMBER WHEN YOU CLONE THE REPOSITORY YOU NEED TO ADD THE VIDEO IN THE FILE NAMED DATA**
**YOU NEED TO HAVE DOCKER DESKTOP INSTALL**
