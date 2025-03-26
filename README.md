# U2-Net Evaluation on ECSSD Dataset

This repository evaluates the U2-Net (and U2NetP) model for salient object detection (SOD) on the ECSSD dataset. 

U2-Net is a deep learning model for salient object detection. This repository runs inference and evaluates the U2Net model on the ECSSD dataset, providing metrics like IoU, Dice score, and F1 score.

## Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

Clone the repository:

```bash
git clone https://github.com/your-repository/U2Net-Salient-Object-Detection.git
cd U2Net-Salient-Object-Detection
```

### Folder Structure

```
U2Net-Salient-Object-Detection/
├── data/
│   ├── ECSSD/
│   │   ├── images/
│   │   └── ground_truth_mask/
├── results/
│   └── ECSSD_results/
├── src/
│   ├── u2net_test.py
│   └── evaluation.py
├── saved_models/
│   └── u2netp/
│       └── u2netp.pth
├── requirements.txt
└── README.md
```

## Inference

Run the following to perform inference:

```bash
cd src
python3 u2net_test.py
```

This will generate predicted saliency maps and save them in `../results/ECSSD_results/`.

### Required Directories:
- `../data/ECSSD/images/`
- `../results/ECSSD_results/`

## Evaluation

To evaluate the model’s performance:

```bash
python3 evaluation.py
```

This will compare predictions with ground truth masks and print the mean IoU, Dice, and F1 scores.

### Required Directories:
- `../data/ECSSD/ground_truth_mask/`
- `../results/ECSSD_results/`

## Metrics

- **IoU**: Measures the overlap between predicted and ground truth masks.
- **Dice Score**: Measures similarity between predicted and ground truth masks.
- **F1 Score**: Balances precision and recall for predicted masks.
