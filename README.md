# Real-Time Touchless Palmprint Biometric Authentication System

A contactless biometric authentication system designed for real-time edge execution. The pipeline captures live video streams, tracks hand positioning via MediaPipe, normalizes mirror-drift coordinates, extracts principal valley ridges using the Modified Finite Radon Transform (MFRAT), and performs rapid cosine similarity verification over 512-dimensional ResNet50 embeddings without requiring relational database overhead.

---

## Key Features

- **Touchless Landmark Tracking:** Uses MediaPipe Hand Tracking with mirror-drift coordinate mapping (\(x \rightarrow w - x\)) to preserve spatial alignment.
- **15-Frame Temporal Stability Filter:** Eliminates motion blur by requiring stable hand placement over consecutive frames before region-of-interest (ROI) cropping.
- **Dedicated Preprocessing Pipeline:** Sequentially chains Data Augmentation → CLAHE → Gaussian Blur → Pixel Normalization → MFRAT texture extraction to isolate ridges under fluctuating ambient lighting.
- **Open-World Verification:** Decouples feature extraction (ResNet50) from authentication (Cosine Similarity), enabling zero-retraining user enrollment and profile deletion.
- **High-Speed Vectorized Matching:** Calculates matrix dot products in active memory via NumPy, completing template comparisons in ≤ 200 ms.
- **Thread-Isolated Desktop UI:** Features an asynchronous CustomTkinter dashboard maintaining a smooth 30 FPS video feed without GUI freezing.
- **Zero-Network Local Registry:** Serializes user templates to an isolated, in-memory mapped `database.pkl` file with atomic cache/disk purging.

---

## Pipeline Architecture

```
Live Camera Stream (30 FPS)
           │
           ▼
MediaPipe Landmark Tracking & Mirror Drift Correction (x → w - x)
           │
           ▼
15-Frame Temporal Stability Check → Upright 224x224 ROI Extraction
           │
           ▼
Sequential Preprocessing Engine:
[Augmentation (10x) → CLAHE → Gaussian Blur → Image Normalization → MFRAT]
           │
           ▼
Feature Projection (Pre-trained ResNet50 → 512-D Feature Vector)
           │
           ▼
Vectorized Cosine Similarity Matching Gate vs. Local Registry (database.pkl)
           │
           ▼
Asynchronous GUI Status Dispatch (CustomTkinter)
```

---

## System Requirements

- **Operating System:** Windows 10/11, macOS (Intel/Apple Silicon), or Linux (Ubuntu 20.04+)
- **Python Version:** Python 3.9 – 3.11 (Python 3.10 recommended)
- **Camera:** Standard USB webcam (720p or 1080p, 30 FPS capable)
- **Hardware:** Minimum 4 GB RAM, dual-core CPU with AVX2 instruction support

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/palmprint-biometric-system.git
cd palmprint-biometric-system
```

### 2. Create and Activate a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
Create a `requirements.txt` file (or use the one provided) and run:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dependency Manifest (`Requirement.txt`)

---

## Project Directory Structure

```text
palmprint-biometric-system/
│
├── core/
│   ├── 01_Augmentation.ipynb        # 10x geometric & illumination variation generator
│   ├── 02_ROI.ipynb           # MediaPipe landmark tracking & mirror inversion
│   ├── 03_Grayscale.ipynb      #Grayscaling
│   ├── 04_clahe.ipynb     # CLAHE
│   ├── 05_blur.ipynb    # Gaussian blur
│   ├── 06_Normalization.ipynb   #Normalization
│   ├── 07_MFRAT.ipynb   #MFRAT
│   ├── 08_Resnet.ipynb   # ResNet50 model wrapper (512-D projection)
│   └── 09_Matching_resnet   #Authentication
│
├── ui/
│   └── Realtime.ipynb             # CustomTkinter threaded interface layout
│
├── data/
│   └── database.pkl           # Local binary registry for enrolled user vectors
├── Requirement.txt           # Project package dependencies
└── README.md                  # Project documentation
```

---

## Running the Application

Launch the main interface by executing:
```bash
python Realtime.ipynb
```

### Operating Instructions:

1. **Enrollment (New User):**
   - Click **Enroll New User** on the dashboard.
   - Enter the user's name/ID.
   - Hold your palm steady inside the screen guides until the 15-frame stability indicator completes.
   - The system captures a snapshot, computes 10 augmented variations, passes them through CLAHE → Gaussian Blur → Normalization → MFRAT → ResNet50, and saves the vectors to `database.pkl`.

2. **Real-Time Verification:**
   - Present your palm to the camera feed.
   - Once tracking locks and the stability window is satisfied, the system extracts the live ROI and performs vectorized Cosine Similarity matching against all profiles.
   - The status banner will update to **ACCESS GRANTED: [User Name]** or **ACCESS DENIED**.

3. **User Management:**
   - Existing profiles can be deleted directly via the interface. Deletion completely purges the associated vector matrix and keys from active memory cache and the binary file simultaneously.

---

## Mathematical Formulation

- **Coordinate Inversion (Mirror Drift Correction):**
  $$x' = w - x$$
  *(where $w$ is the frame width, e.g., 640 px)*

- **Cosine Similarity Verification Gate:**
  $$\text{Similarity}(A, B) = \frac{A \cdot B}{\Vert{}A\Vert{} \Vert{}B\Vert{}} = \frac{\sum_{i=1}^{512} A_i B_i}{\sqrt{\sum_{i=1}^{512} A_i^2} \sqrt{\sum_{i=1}^{512} B_i^2}}$$

- **Matching Rule:**
  $$\text{Decision} = \begin{cases} \text{Grant Access}, & \max(\text{Similarity}) \ge \tau \\ \text{Deny Access}, & \max(\text{Similarity}) < \tau \end{cases}$$
  *(where $\tau$ is the configurable verification threshold, default: `0.80`)*

---

## Troubleshooting

- **Webcam Access Denied:** Ensure the terminal/IDE has permission to access system camera devices.
- **Low FPS / Frame Drop:** Ensure OpenCV is using default hardware acceleration. Check that background inferences are running asynchronously without blocking the main CustomTkinter GUI loop.
- **Corrupted Registry:** If `database.pkl` becomes unreadable, back it up and delete it; the application will automatically reinitialize an empty template dictionary on the next start.