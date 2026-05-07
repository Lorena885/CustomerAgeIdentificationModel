# Customer Age Identification Model

Computer vision project for estimating customer age from facial images using a TensorFlow/Keras regression model with a pretrained ResNet50 backbone.

The model was trained on 7,591 labeled facial images and predicts age as a continuous numeric value. The final validation performance reached an MAE of approximately **8.91 years**.

## Project Overview

This project explores whether deep learning can estimate a person's age from facial images with a practical level of accuracy. It uses transfer learning with ResNet50, image preprocessing, batch-based data loading, and early stopping to train an age regression model efficiently.

The repository includes:

- A complete training notebook in Spanish.
- An English portfolio notebook.
- The trained Keras model.
- The labeled image dataset.
- A static HTML dashboard summarizing the project and results.

## Repository Structure

```text
CustomerAgeIdentificationModel/
+-- final_files/                     # Facial image dataset
+-- age_model.keras                  # Trained Keras model
+-- index.html                       # Static dashboard / portfolio page
+-- labels.csv                       # Image filenames and real ages
+-- notebook.ipynb                   # Main notebook in Spanish
+-- notebook_english_portfolio.ipynb # English portfolio version
+-- README.md                        # Project documentation
```

## Dataset

The dataset contains:

- **7,591 images**
- **7,591 labels**
- No missing image files
- No missing label values
- No duplicated filenames

The label file contains two columns:

| Column | Description |
| --- | --- |
| `file_name` | Image filename inside `final_files/` |
| `real_age` | Real age associated with the image |

The age range goes from **1 to 100 years**, with a mean age of approximately **31.2 years**.

## Methodology

The model pipeline includes:

1. Loading image labels from `labels.csv`.
2. Creating train and validation generators with `ImageDataGenerator`.
3. Applying ResNet50-compatible preprocessing.
4. Splitting the dataset into 80% training and 20% validation.
5. Using a pretrained ResNet50 backbone with frozen convolutional layers.
6. Adding a regression head for continuous age prediction.
7. Training with early stopping based on validation MAE.
8. Saving the trained model as `age_model.keras`.

## Model Architecture

The model uses:

- `ResNet50(weights="imagenet", include_top=False)`
- `GlobalAveragePooling2D`
- `Dense(128, activation="relu")`
- `Dropout(0.2)`
- `Dense(1)` for age regression

Training configuration:

- Optimizer: Adam
- Learning rate: `1e-4`
- Loss: Mean Squared Error
- Metric: Mean Absolute Error
- Early stopping monitor: `val_mae`

## Results

The trained model achieved:

| Metric | Value |
| --- | --- |
| Training images | 6,073 |
| Validation images | 1,518 |
| Final validation MAE | ~8.91 years |
| Model parameters | ~23.85M |

This means that, on average, the model prediction differs from the real age by about nine years on the validation set.

## How to Run

Create and activate a Python environment, then install the main dependencies:

```bash
pip install pandas numpy matplotlib tensorflow
```

Open either notebook:

- `notebook.ipynb` for the Spanish version
- `notebook_english_portfolio.ipynb` for the English version

Then run the notebook cells in order.

The expected local dataset path is:

```python
path = "./"
```

The notebook expects:

```text
labels.csv
final_files/
```

to be located in the repository root.

## Dashboard

The project includes a static dashboard in `index.html` with a visual summary of the dataset, model architecture, training curves, and final results.

If GitHub Pages is enabled for this repository, the dashboard can be published directly from this file.

## Key Takeaways

- Transfer learning with ResNet50 is effective for extracting facial image features.
- Batch-based image loading avoids loading the full dataset into memory.
- The dataset is usable and clean, but age distribution is not perfectly balanced.
- Model performance should be interpreted as approximate age estimation, not exact identification.
- The model can support customer age range analysis, demographic exploration, or visual verification workflows.

## Limitations

- Age estimation from facial images is inherently uncertain.
- Performance may vary across underrepresented age groups.
- The model should not be used as the sole decision-making tool in sensitive or high-stakes contexts.
- Additional fairness and bias evaluation would be recommended before production use.

## Technologies

- Python
- TensorFlow / Keras
- ResNet50
- Pandas
- NumPy
- Matplotlib
- HTML / CSS / JavaScript
- Chart.js

## Author

Developed as a computer vision portfolio project for customer age estimation from facial images.
