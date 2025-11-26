## 🌍 Earthquake & Seismic Events Dashboard

## 🔍 Overview
**Earthquake & Seismic Events Dashboard** is a beautiful, interactive web dashboard that visualizes and analyzes global earthquake and nuclear explosion data from **1965 to 2016**. Built with **Python (Flask)**, **Matplotlib**, **Seaborn**, and **Folium**, this project transforms raw seismic data into insightful, publication-quality visualizations — including frequency trends, energy release patterns, depth-magnitude relationships, aftershock sequences, and a global map of nuclear tests.
Perfect for seismologists, researchers, students, or anyone curious about Earth's seismic history.

## 📸 Screenshots
![Earthquake Dashboard Preview](screenshot.png)

## ✨ Features
  - ✅ Decadal Earthquake Frequency Trend – See how reported earthquakes increased over time
  - ✅ Cumulative Seismic Energy Release – Visualize the exponential impact of major events
  - ✅ Magnitude vs. Depth Analysis – Box plots and bubble charts revealing seismicity patterns
  - ✅ Gutenberg-Richter Law Verification – Log-linear frequency-magnitude relationship
  - ✅ 2011 Japan (Tōhoku) Aftershock Sequence – 30-day case study of one of history’s largest quakes
  - ✅ Nuclear Explosion Global Map – Interactive Folium map showing all recorded nuclear tests (1965–2016)
  - ✅ Reporting Agency Comparison – Violin plots comparing magnitude distributions by source
  - ✅ Fully Responsive Design – Clean Tailwind CSS layout that works on desktop and mobile
  - ✅ Zero JavaScript Required – Pure server-side rendering with Flask
    
## 🧠 How It Works
  - The app uses **Flask** as a lightweight web server to serve a dynamic HTML dashboard
  - All plots are generated on-the-fly using **Matplotlib** and **Seaborn**, then embedded as base64-encoded PNGs
  - An interactive **Folium** map displays nuclear explosion locations with popups
  - Data is loaded from a single `earthquakes.csv` file (from the USGS/significant earthquakes dataset)
  - Each visualization is accompanied by clear titles, labels, and scientific context
    
## 🛠️ Built With
  - Python 3 – Core backend logic
  - Flask – Web framework and templating
  - Pandas – Data manipulation and analysis
  - Matplotlib & Seaborn – Publication-quality static visualizations
  - Folium – Interactive leaflet maps for nuclear explosion events
  - Tailwind CSS (via CDN) – Modern, responsive styling
  - Jinja2 – Dynamic HTML templating
    
## 🧰 Getting Started
To run the Earthquake Dashboard locally:
### Prerequisites
- Python 3.8+
- `earthquakes.csv` file in the project root (download from:
  → https://earthquake.usgs.gov/earthquakes/search/ or use the standard significant earthquakes dataset)
### Installation & Run
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/earthquake-seismic-dashboard.git
   ```
2. **Navigate to the Project Directory:**
  ```bash
  cd earthquake-seismic-dashboard
  ```
3. **Install Dependencies:**
   ```bash
   pip install flask pandas matplotlib seaborn folium
   ```
4. **Place the Dataset:**
  Download the earthquake dataset and name it earthquakes.csv
  Place it in the same folder as app.py
5. **Run the App:**
   ```bash
   python app.py
   ```
6. **Open Your Browser:**Go to → http://127.0.0.1:5000
You now have a fully functional seismic analysis dashboard running locally!

## 📊 Dataset Source
  - The dashboard uses the USGS Significant Earthquakes Archive (1965–2016)
  - Link: https://earthquake.usgs.gov/earthquakes/search/
    (Includes columns: Date, Time, Latitude, Longitude, Depth, Magnitude, Type, Source, etc.)

## 🧪 Testing
  - Tested on Python 3.9–3.12
  - Works on Windows, macOS, and Linux
  - Verified rendering in Chrome, Firefox, Edge, and Safari
  - Responsive design confirmed on mobile and tablet screens

## 📖 What I Learned
  - Generating and embedding dynamic Matplotlib plots into web pages using base64
  - Creating interactive geographic visualizations with Folium and Flask
  - Cleaning and transforming large real-world seismic datasets with Pandas
  - Applying classic seismology concepts: Gutenberg-Richter law, aftershock decay, energy scaling
  - Building a clean, professional data dashboard using Flask + Tailwind CSS without JavaScript frameworks
  - Balancing scientific accuracy with visually appealing design

## 🤝 Contributing
Contributions are welcome\! If you have ideas for new features or improvements, feel free to fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.
