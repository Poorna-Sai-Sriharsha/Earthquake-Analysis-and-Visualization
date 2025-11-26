import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from flask import Flask, render_template
import io
import base64
app = Flask(__name__)

def generate_plots():
    try:
        df = pd.read_csv('earthquakes.csv')
    except FileNotFoundError:
        return {"error": "The 'earthquakes.csv' file was not found. Please make sure it's in the same directory as app.py."}

    # --- Data Preparation ---
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Year'] = df['Date'].dt.year
    df['Decade'] = (df['Year'] // 10) * 10
    df['Hour'] = df['Date'].dt.hour
    
    plots = {}

    # --- Plot 1: Earthquake Frequency by Decade ---
    plt.figure(figsize=(12, 6))
    dc = df['Decade'].value_counts().sort_index()
    sns.lineplot(x=dc.index, y=dc.values, marker='o', color='royalblue')
    plt.title('Earthquake Frequency by Decade (1965-2016)', fontsize=16)
    plt.xlabel('Decade')
    plt.ylabel('Number of Earthquakes')
    plt.grid(True)
    plt.xticks(dc.index)
    plots['plot1'] = save_plot_to_base64()

    # --- Plot 2: Cumulative Seismic Energy Release ---
    df['Energy'] = 10**(1.5 * df['Magnitude'])
    df.sort_values('Date', inplace=True)
    df['Cumulative Energy'] = df['Energy'].cumsum()
    top_energy_events = df.sort_values(by='Energy', ascending=False).head(4)
    plt.figure(figsize=(12, 7))
    plt.plot(df['Date'], df['Cumulative Energy'], color='red')
    plt.title('Cumulative Seismic Energy Release (1965-2016)', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Cumulative Energy Released (Relative Scale)')
    plt.grid(True)
    for _, row in top_energy_events.iterrows():
        plt.axvline(row['Date'], color='gray', linestyle='--', lw=0.5)
        plt.text(row['Date'], row['Cumulative Energy'], f" {row['Date'].year}\nM {row['Magnitude']} ", va='top')
    plots['plot2'] = save_plot_to_base64()

    # --- Plot 3: Magnitude by Depth Category ---
    def assign_depth_category(depth):
        if depth < 70: return 'Shallow'
        elif 70 <= depth <= 300: return 'Intermediate'
        else: return 'Deep'
    df['Depth Category'] = df['Depth'].apply(assign_depth_category)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Depth Category', y='Magnitude', data=df, order=['Shallow', 'Intermediate', 'Deep'], palette='viridis')
    plt.title('Magnitude Distribution by Depth Category', fontsize=16)
    plt.xlabel('Depth Category')
    plt.ylabel('Magnitude')
    plt.grid(True, axis='y')
    plots['plot3'] = save_plot_to_base64()

    # --- Plot 4: Gutenberg-Richter Law ---
    mag_counts = df['Magnitude'].round(1).value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    plt.plot(mag_counts.index, mag_counts.values, 'o-', color='crimson')
    plt.title('Gutenberg-Richter Law: Frequency vs. Magnitude', fontsize=16)
    plt.xlabel('Magnitude')
    plt.ylabel('Number of Earthquakes (Log Scale)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plots['plot4'] = save_plot_to_base64()

    # --- Plot 5: Japan Aftershock Sequence ---
    mainshock_date = pd.to_datetime('2011-03-11')
    aftershocks = df[
        (df['Date'] >= mainshock_date) &
        (df['Date'] < mainshock_date + pd.Timedelta(days=30)) &
        (df['Latitude'] >= 33.322) & (df['Latitude'] <= 43.322) &
        (df['Longitude'] >= 137.369) & (df['Longitude'] <= 147.369) &
        (df['Magnitude'] < 9.0)
    ].copy()
    plt.figure(figsize=(15, 7))
    plt.scatter(aftershocks['Date'], aftershocks['Magnitude'], alpha=0.5, color='royalblue', label='Aftershock')
    plt.axvline(x=mainshock_date, color='red', linestyle='--', linewidth=2, label='Mainshock (M 9.1)')
    plt.title('Aftershock Sequence of the 2011 Tōhoku, Japan Earthquake (30 Days)', fontsize=16)
    plt.xlabel('Date (March-April 2011)', fontsize=12)
    plt.ylabel('Aftershock Magnitude', fontsize=12)
    plt.ylim(4.5, 8.0)
    plt.legend()
    plots['plot5'] = save_plot_to_base64()
    
    # --- Plot 7: Magnitude vs. Depth Bubble Chart ---
    magnitude_bins = [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.5]
    magnitude_labels = [f'{magnitude_bins[i]} - {magnitude_bins[i+1]}' for i in range(len(magnitude_bins)-1)]
    df['Magnitude Bin'] = pd.cut(df['Magnitude'], bins=magnitude_bins, labels=magnitude_labels, right=False)
    bubble_data = df.groupby(['Depth Category', 'Magnitude Bin']).size().reset_index(name='Count')
    bubble_data = bubble_data[bubble_data['Count'] > 0]
    plt.figure(figsize=(14, 8))
    bubble_plot = sns.scatterplot(
        data=bubble_data, x='Magnitude Bin', y='Depth Category', size='Count',
        hue='Count', palette='plasma', sizes=(50, 2000), legend='auto'
    )
    plt.title('Hotspots of Earthquake Activity: Depth vs. Magnitude', fontsize=18)
    plt.xlabel('Magnitude Range', fontsize=12)
    plt.ylabel('Depth Category', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    h, l = bubble_plot.get_legend_handles_labels()
    plt.legend(h[0:7], l[0:7], title="Number of Earthquakes", bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plots['plot7'] = save_plot_to_base64()

    # --- Plot 8: Magnitude by Reporting Agency ---
    top_sources = df['Source'].value_counts().head(5).index
    df_top_sources = df[df['Source'].isin(top_sources)]
    plt.figure(figsize=(12, 7))
    sns.violinplot(x='Source', y='Magnitude', data=df_top_sources, palette='plasma')
    plt.title('Magnitude Distribution for Top 5 Reporting Agencies', fontsize=16)
    plt.xlabel('Reporting Agency (Source)')
    plt.ylabel('Magnitude')
    plots['plot8'] = save_plot_to_base64()

    nuclear_df = df[df['Type'] == 'Nuclear Explosion'].copy()
    nuclear_map = folium.Map(location=[49.8, 78.5], zoom_start=3, tiles="CartoDB dark_matter")
    for _, row in nuclear_df.iterrows():
        popup_text = f"<b>Event Type:</b> {row['Type']}<br><b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br><b>Magnitude:</b> {row['Magnitude']}"
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']], radius=row['Magnitude'] * 1.5, color='#ff3300',
            fill=True, fill_opacity=0.6, popup=folium.Popup(popup_text, max_width=300)
        ).add_to(nuclear_map)
    plots['map10'] = nuclear_map._repr_html_()

    return plots

def save_plot_to_base64():
    """Saves the current matplotlib plot to a base64 encoded string."""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close() # Close the plot to free up memory
    return plot_url

@app.route('/')
def dashboard():
    """Renders the dashboard page with all the plots."""
    plot_data = generate_plots()
    if "error" in plot_data:
        # A simple error page if the CSV is missing
        return f"<p style='font-family: sans-serif; color: red;'>Error: {plot_data['error']}</p>"
    return render_template('index.html', plots=plot_data)

if __name__ == '__main__':
    app.run(debug=True)
