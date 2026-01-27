"""
IoT System Architecture Diagram Generator
Smart Water Quality Monitoring System
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.set_aspect('equal')
ax.axis('off')

# Color scheme
colors = {
    'sensors': '#3498db',      # Blue
    'edge': '#2ecc71',         # Green
    'network': '#9b59b6',      # Purple
    'cloud': '#e74c3c',        # Red
    'ml': '#f39c12',           # Orange
    'dashboard': '#1abc9c',    # Teal
    'text': '#2c3e50',         # Dark blue-gray
    'arrow': '#7f8c8d',        # Gray
    'bg_light': '#ecf0f1',     # Light gray
    'white': '#ffffff'
}

def draw_rounded_box(ax, x, y, width, height, color, title, items=None, alpha=0.9):
    """Draw a rounded rectangle with title and optional bullet points."""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05,rounding_size=0.3",
                         facecolor=color, edgecolor='white', linewidth=2, alpha=alpha)
    ax.add_patch(box)
    
    # Title
    ax.text(x + width/2, y + height - 0.3, title,
            ha='center', va='top', fontsize=11, fontweight='bold', color='white')
    
    # Items
    if items:
        for i, item in enumerate(items):
            ax.text(x + 0.2, y + height - 0.7 - i*0.35, f"• {item}",
                   ha='left', va='top', fontsize=8, color='white')

def draw_arrow(ax, start, end, color='#7f8c8d', style='->'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                              connectionstyle='arc3,rad=0'))

def draw_data_flow_arrow(ax, start, end, label=''):
    """Draw a data flow arrow with optional label."""
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=colors['arrow'], lw=3,
                              mutation_scale=20))
    if label:
        ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom',
               fontsize=8, color=colors['text'], style='italic')

# Title
ax.text(8, 11.5, 'IoT Water Quality Monitoring System Architecture',
        ha='center', va='center', fontsize=18, fontweight='bold', color=colors['text'])
ax.text(8, 11.0, 'Smart Water Quality Monitoring Using IoT & Machine Learning',
        ha='center', va='center', fontsize=10, color=colors['arrow'], style='italic')

# ============= LAYER 1: SENSORS =============
sensor_x, sensor_y = 0.5, 6.5
sensor_w, sensor_h = 3, 4

# Main sensor box
draw_rounded_box(ax, sensor_x, sensor_y, sensor_w, sensor_h, colors['sensors'],
                'SENSOR LAYER', 
                ['Turbidity Sensor', 'Conductivity Sensor', 'Temperature Sensor',
                 'NO₃ Sensor', 'Water Level Sensor', 'Flow Rate Sensor'])

# Sensor icons (small circles representing sensors)
for i, (name, y_off) in enumerate([('T', 0), ('C', 0.5), ('Temp', 1)]):
    circle = Circle((sensor_x + 2.7, sensor_y + 1.2 + y_off*0.8), 0.15,
                   facecolor='white', edgecolor=colors['sensors'], linewidth=2)
    ax.add_patch(circle)

# Label
ax.text(sensor_x + sensor_w/2, sensor_y - 0.3, '11 Monitoring Stations\nQueensland Rivers',
        ha='center', va='top', fontsize=8, color=colors['text'])

# ============= LAYER 2: EDGE PROCESSING =============
edge_x, edge_y = 4.5, 6.5
edge_w, edge_h = 3, 4

draw_rounded_box(ax, edge_x, edge_y, edge_w, edge_h, colors['edge'],
                'EDGE PROCESSING',
                ['Data Validation', 'Noise Filtering', 'Temporal Aggregation',
                 'Local Alerting', 'Data Buffering', '(7-day capacity)'])

# ============= LAYER 3: NETWORK =============
net_x, net_y = 8.5, 6.5
net_w, net_h = 3, 4

draw_rounded_box(ax, net_x, net_y, net_w, net_h, colors['network'],
                'NETWORK LAYER',
                ['MQTT Protocol', 'TLS 1.3 Encryption', '4G LTE (Primary)',
                 'LoRaWAN (Backup)', 'QoS Level 1'])

# ============= LAYER 4: CLOUD PLATFORM =============
cloud_x, cloud_y = 12.5, 6.5
cloud_w, cloud_h = 3, 4

draw_rounded_box(ax, cloud_x, cloud_y, cloud_w, cloud_h, colors['cloud'],
                'CLOUD PLATFORM',
                ['Data Ingestion', 'Time Series DB', 'Data Lake Storage',
                 'Stream Processing', 'Alert Service'])

# ============= ARROWS BETWEEN MAIN LAYERS =============
arrow_y = 8.5
draw_data_flow_arrow(ax, (sensor_x + sensor_w, arrow_y), (edge_x, arrow_y), '10s samples')
draw_data_flow_arrow(ax, (edge_x + edge_w, arrow_y), (net_x, arrow_y), 'Hourly data')
draw_data_flow_arrow(ax, (net_x + net_w, arrow_y), (cloud_x, arrow_y), 'JSON/MQTT')

# ============= ML PROCESSING LAYER =============
ml_x, ml_y = 5.5, 2
ml_w, ml_h = 5, 3

# ML background box
ml_bg = FancyBboxPatch((ml_x - 0.2, ml_y - 0.2), ml_w + 0.4, ml_h + 0.4,
                       boxstyle="round,pad=0.05,rounding_size=0.3",
                       facecolor=colors['bg_light'], edgecolor=colors['ml'], 
                       linewidth=2, alpha=0.5)
ax.add_patch(ml_bg)

ax.text(ml_x + ml_w/2, ml_y + ml_h + 0.1, 'MACHINE LEARNING LAYER',
        ha='center', va='bottom', fontsize=11, fontweight='bold', color=colors['ml'])

# LSTM Model Box
lstm_x, lstm_y = ml_x + 0.2, ml_y + 0.3
lstm_w, lstm_h = 2.2, 2.4

lstm_box = FancyBboxPatch((lstm_x, lstm_y), lstm_w, lstm_h,
                          boxstyle="round,pad=0.03,rounding_size=0.2",
                          facecolor=colors['ml'], edgecolor='white', linewidth=2)
ax.add_patch(lstm_box)

ax.text(lstm_x + lstm_w/2, lstm_y + lstm_h - 0.2, 'LSTM Model',
        ha='center', va='top', fontsize=10, fontweight='bold', color='white')
ax.text(lstm_x + lstm_w/2, lstm_y + lstm_h - 0.5, 'Turbidity\nPrediction',
        ha='center', va='top', fontsize=9, color='white')
ax.text(lstm_x + lstm_w/2, lstm_y + 0.6, '48h input → 24h forecast',
        ha='center', va='center', fontsize=7, color='white', style='italic')
ax.text(lstm_x + lstm_w/2, lstm_y + 0.25, 'R² > 0.80',
        ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# Random Forest Model Box
rf_x, rf_y = ml_x + 2.6, ml_y + 0.3
rf_w, rf_h = 2.2, 2.4

rf_box = FancyBboxPatch((rf_x, rf_y), rf_w, rf_h,
                        boxstyle="round,pad=0.03,rounding_size=0.2",
                        facecolor=colors['ml'], edgecolor='white', linewidth=2)
ax.add_patch(rf_box)

ax.text(rf_x + rf_w/2, rf_y + rf_h - 0.2, 'Random Forest',
        ha='center', va='top', fontsize=10, fontweight='bold', color='white')
ax.text(rf_x + rf_w/2, rf_y + rf_h - 0.5, 'Classification',
        ha='center', va='top', fontsize=9, color='white')
ax.text(rf_x + rf_w/2, rf_y + 0.6, 'Safe/Warning/Unsafe',
        ha='center', va='center', fontsize=7, color='white', style='italic')
ax.text(rf_x + rf_w/2, rf_y + 0.25, 'Accuracy: 94%',
        ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# Arrow from Cloud to ML
draw_data_flow_arrow(ax, (cloud_x + cloud_w/2, cloud_y), (ml_x + ml_w/2, ml_y + ml_h + 0.5), '')

# ============= DASHBOARD/OUTPUT LAYER =============
dash_x, dash_y = 12, 2
dash_w, dash_h = 3.5, 3

draw_rounded_box(ax, dash_x, dash_y, dash_w, dash_h, colors['dashboard'],
                'VISUALIZATION',
                ['Tableau Dashboard', 'Real-time Status', 'Predictions Display',
                 'Alert Notifications', 'Historical Trends'])

# Arrow from ML to Dashboard
draw_data_flow_arrow(ax, (ml_x + ml_w, ml_y + ml_h/2), (dash_x, dash_y + dash_h/2), 'Predictions')

# ============= BOTTOM INFO BOXES =============
# Data stats box
stats_x, stats_y = 0.5, 2
stats_w, stats_h = 4.5, 3

stats_bg = FancyBboxPatch((stats_x, stats_y), stats_w, stats_h,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=colors['bg_light'], edgecolor=colors['text'], 
                          linewidth=1.5, alpha=0.7)
ax.add_patch(stats_bg)

ax.text(stats_x + stats_w/2, stats_y + stats_h - 0.2, 'Dataset Statistics',
        ha='center', va='top', fontsize=10, fontweight='bold', color=colors['text'])

stats_text = [
    '• Total Records: 295,754',
    '• Stations: 11',
    '• Date Range: 2016-2020',
    '• Sampling: Hourly',
    '• Features: 41 (engineered)'
]
for i, text in enumerate(stats_text):
    ax.text(stats_x + 0.3, stats_y + stats_h - 0.6 - i*0.4, text,
           ha='left', va='top', fontsize=8, color=colors['text'])

# ============= LEGEND / THRESHOLDS =============
thresh_x, thresh_y = 0.5, 0.3
ax.text(thresh_x, thresh_y + 1.2, 'Water Quality Thresholds:', 
        fontsize=9, fontweight='bold', color=colors['text'])

# Safe indicator
safe_box = FancyBboxPatch((thresh_x, thresh_y + 0.7), 0.8, 0.35,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='#27ae60', edgecolor='white', linewidth=1)
ax.add_patch(safe_box)
ax.text(thresh_x + 0.4, thresh_y + 0.87, 'Safe', ha='center', va='center', 
        fontsize=8, color='white', fontweight='bold')
ax.text(thresh_x + 1.0, thresh_y + 0.87, 'Turbidity < 5 NTU', ha='left', va='center', 
        fontsize=7, color=colors['text'])

# Warning indicator
warn_box = FancyBboxPatch((thresh_x, thresh_y + 0.3), 0.8, 0.35,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='#f39c12', edgecolor='white', linewidth=1)
ax.add_patch(warn_box)
ax.text(thresh_x + 0.4, thresh_y + 0.47, 'Warning', ha='center', va='center', 
        fontsize=8, color='white', fontweight='bold')
ax.text(thresh_x + 1.0, thresh_y + 0.47, 'Turbidity 5-50 NTU', ha='left', va='center', 
        fontsize=7, color=colors['text'])

# Unsafe indicator
unsafe_box = FancyBboxPatch((thresh_x, thresh_y - 0.1), 0.8, 0.35,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor='#e74c3c', edgecolor='white', linewidth=1)
ax.add_patch(unsafe_box)
ax.text(thresh_x + 0.4, thresh_y + 0.07, 'Unsafe', ha='center', va='center', 
        fontsize=8, color='white', fontweight='bold')
ax.text(thresh_x + 1.0, thresh_y + 0.07, 'Turbidity > 50 NTU', ha='left', va='center', 
        fontsize=7, color=colors['text'])

# ============= PROTOCOL INFO =============
proto_x = 4.5
ax.text(proto_x, thresh_y + 1.2, 'Communication Protocol:', 
        fontsize=9, fontweight='bold', color=colors['text'])
proto_items = ['MQTT v3.1.1 + TLS 1.3', 'Primary: 4G LTE Cellular', 
               'Backup: LoRaWAN', 'QoS Level 1 (at least once)']
for i, item in enumerate(proto_items):
    ax.text(proto_x, thresh_y + 0.8 - i*0.35, f'• {item}',
           ha='left', va='top', fontsize=7, color=colors['text'])

# ============= FOOTER =============
ax.text(8, -0.3, 'AAI-530 Final Project | Smart Water Quality Monitoring System | Queensland, Australia',
        ha='center', va='center', fontsize=8, color=colors['arrow'], style='italic')

# Save the figure
plt.tight_layout()
plt.savefig('../diagrams/iot_system_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('../diagrams/iot_system_architecture.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Diagram saved to: diagrams/iot_system_architecture.png")
print("Diagram saved to: diagrams/iot_system_architecture.pdf")
print("Done!")
