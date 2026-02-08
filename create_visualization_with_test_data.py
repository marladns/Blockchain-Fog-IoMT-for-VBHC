import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample data creation
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)

def create_visualization(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data['x'], data['y'], label='Sine Wave with Noise')
    plt.title('Visualization of Test Data')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.legend()
    plt.grid()
    plt.savefig('visualization.png')
    plt.show()


def validate_data(data):
    if data.isnull().values.any():
        raise ValueError('Data contains null values.')
    if not isinstance(data['x'], np.ndarray) or not isinstance(data['y'], np.ndarray):
        raise TypeError('Data is not in the correct format.')

# Creating test data
data = pd.DataFrame({'x': x, 'y': y})

# Validation
try:
    validate_data(data)
    create_visualization(data)
    print('Data validated and visualization created successfully.')
except Exception as e:
    print(f'Error: {e}')