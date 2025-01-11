<h1 align="center">TrueColors - Python Backend<h1>

TrueColors backend is a Python API responsible for processing images, simulating how they would appear to color-blind individuals, and detecting objects in images using an AI model. The API handles requests from the React Native frontend to process images for color-blind simulations and AI object detection.

## Features

- **Color Blindness Simulation**: Processes images and simulates how they would appear to people with Protanopia, Deuteranopia, or Tritanopia.
- **Color Enhancement**: Enhances images for color-blind users to help them see colors as they would appear to individuals with normal vision.
- **AI Object Detection**: Detects objects in images and returns their names along with their colors.
- **BLIP-based AI Model**: Utilizes a pre-trained BLIP model for image captioning and object detection.

## Installation

### Prerequisites

Ensure that you have the following installed:

- Python 3.x
- Pip (Python package manager)
- Virtualenv (optional but recommended)

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/qrivnel/truecolors-python-backend.git
cd truecolors-python-backend