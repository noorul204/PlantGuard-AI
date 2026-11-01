\# PlantGuard-AI 🌿



\### AI-Powered Plant Disease Detection



PlantGuard-AI is a machine learning project that analyzes images of potato and tomato leaves to classify them as healthy or affected by early blight. It combines a deep learning model with a Flask backend and a simple web interface.



\## Features



\- Upload a potato or tomato leaf image.

\- Preview the selected image before prediction.

\- Classify the image into one of four categories.

\- Display the predicted class and confidence score.

\- Show information about the predicted condition.

\- Display common symptoms and prevention suggestions.

\- Handle missing or invalid image uploads.



\## Technologies Used



\- \*\*Python\*\* — application and model development

\- \*\*TensorFlow and Keras\*\* — deep learning

\- \*\*MobileNetV2\*\* — pretrained image classification model

\- \*\*Flask\*\* — backend web framework

\- \*\*HTML, CSS and JavaScript\*\* — frontend

\- \*\*NumPy and Pillow\*\* — image preprocessing



\## Supported Classes



The model currently supports four classes:



1\. Potato — Early Blight

2\. Potato — Healthy

3\. Tomato — Early Blight

4\. Tomato — Healthy



\## How It Works



1\. The user uploads a plant-leaf image.

2\. The frontend sends the image to the Flask backend.

3\. The backend resizes the image to 224 × 224 pixels and prepares it for prediction.

4\. The trained deep learning model predicts the class.

5\. The backend returns the predicted class, confidence score, symptoms, and prevention information.

6\. The frontend displays the result to the user.



\## Model Training and Evaluation



The model uses MobileNetV2 with pretrained ImageNet weights. The base model is frozen, and additional classification layers are trained for the four supported classes.



The model was evaluated using 128 validation images.



\- \*\*Validation accuracy:\*\* 99.22%

\- \*\*Validation loss:\*\* 0.0571



These results apply to the validation dataset used during development. They do not guarantee the same accuracy on images collected in real-world conditions.



\## Project Structure



```text

PlantGuard-AI/

├── app.py

├── predict\_plant.py

├── train\_plant\_model\_v2.py

├── models/

│   └── plant\_disease\_model\_v2.keras

├── templates/

│   └── index.html

├── plant\_data/

│   ├── train/

│   └── validation/

└── README.md



Other dataset files and supporting scripts may also be present in the project folder.



\## How to Run the Project



\### Prerequisites



\- Python 3.13

\- pip

\- The trained model file



\### 1. Open the project directory



```bash

cd /mnt/c/Users/ELCOT/Desktop/PlantGuard-AI

```



\### 2. Activate the Python virtual environment



```bash

source \~/plantguard\_env/bin/activate

```



\### 3. Install the required packages



```bash

python -m pip install tensorflow flask pillow numpy

```



\### 4. Start the application



```bash

python app.py

```



\### 5. Open the website



Open the following address in your browser:



http://127.0.0.1:5000/



Upload a supported plant-leaf image and click \*\*Predict Plant Health\*\* to view the result.



\## Limitations



\- The model supports only the four classes listed above.

\- It may misclassify images with poor lighting, unusual backgrounds, or unfamiliar symptoms.

\- A high confidence score does not guarantee a correct diagnosis.

\- The application is an educational project and should not replace expert agricultural advice.



\## Future Improvements



\- Add more plant species and disease categories.

\- Train and evaluate the model on a larger, more diverse dataset.

\- Improve image validation and error handling.

\- Deploy the application online.

\- Improve the user interface and mobile responsiveness.



\## Disclaimer



PlantGuard-AI provides experimental AI predictions for educational purposes. Always verify disease-related results before making plant-treatment decisions.



