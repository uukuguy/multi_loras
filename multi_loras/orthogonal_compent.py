#!/usr/bin/env python
"""
To achieve the goal of combining two fine-tuned models (A1 and A2) that specialize in different tasks (code and summarization, respectively), with a shared base model (A), and minimize interference between the two sets of weights, you can indeed use an orthogonal approach. The idea here is to isolate the task-specific variations in each model and combine them in a way that preserves their unique strengths.

Here's a step-by-step approach to achieve this:

- Extract Weights: Extract the weights from models A1, A2, and the base model A.
- Calculate Weight Differences: Compute the differences between the weights of A1 and A, and between A2 and A. These differences represent the task-specific adaptations for code and summarization, respectively.
- Isolate Task-Specific Variations: For each set of differences, calculate the orthogonal component with respect to the base model A's weights. This step helps in isolating the task-specific variations.
- Combine Orthogonal Components: Combine the orthogonal components from both A1 and A2. Since these components are orthogonal to the base model's weights, combining them should minimize interference.
- Create New Model: Apply these combined weights to a new model that has the same architecture as A, A1, and A2.
"""
import numpy as np


def calculate_orthogonal_component(base_weights, modified_weights):
    """
    To identify the part of a fine-tuned model's weights that has the greatest impact on the source model weights using an orthogonal approach, you'll need to compare the weights of the fine-tuned model with those of the original (source) model. The goal is to quantify the changes in the weights due to fine-tuning and identify those changes that are most significant.

    Here's a step-by-step approach to do this:

    - Obtain Weights of Both Models: Extract the weights of both the source model and the fine-tuned model. This can typically be done using the model's API in frameworks like TensorFlow or PyTorch.
    - Calculate Weight Differences: Compute the difference between the weights of the fine-tuned model and the source model. This gives you the changes made during fine-tuning.
    - Flatten and Normalize Weight Differences: Flatten these differences into vectors (if they are not already) and normalize them. This step is crucial to treat the weights uniformly regardless of their original shape.
    - Compute Orthogonal Components: For each layer (or set of weights), calculate the orthogonal component of the fine-tuned model's weight differences with respect to the source model's weights. This step will highlight the components of the weight changes that are most orthogonal to the original weights, indicating areas of greatest change or deviation.
    - Analyze and Interpret Results: The orthogonal components with the largest magnitudes represent the parts of the model that have changed the most due to fine-tuning. These are the areas you're interested in.
    """
    weight_difference = modified_weights - base_weights
    weight_difference_flat = weight_difference.flatten()
    base_weights_flat = base_weights.flatten()

    # Normalizing the weight differences
    weight_difference_norm = weight_difference_flat / np.linalg.norm(weight_difference_flat)
    base_weights_norm = base_weights_flat / np.linalg.norm(base_weights_flat)

    # Compute dot product and orthogonal component
    dot_product = np.dot(weight_difference_norm, base_weights_norm)
    orthogonal_component = weight_difference_norm - dot_product * base_weights_norm

    # Reshape back to original shape
    return orthogonal_component.reshape(base_weights.shape)


def combine_models(base_model_weights, model_a1_weights, model_a2_weights):
    # Calculate orthogonal components for each specialization
    orthogonal_a1 = calculate_orthogonal_component(base_model_weights, model_a1_weights)
    orthogonal_a2 = calculate_orthogonal_component(base_model_weights, model_a2_weights)

    # Combine the orthogonal components
    combined_weights = base_model_weights + orthogonal_a1 + orthogonal_a2

    return combined_weights


def evaluate_interference(orthogonal_a1, orthogonal_a2):
    # Flatten the weight matrices
    flat_a1 = orthogonal_a1.flatten()
    flat_a2 = orthogonal_a2.flatten()

    # Normalize the vectors for cosine similarity
    norm_a1 = flat_a1 / np.linalg.norm(flat_a1)
    norm_a2 = flat_a2 / np.linalg.norm(flat_a2)

    # Dot product (Cosine similarity)
    dot_product = np.dot(norm_a1, norm_a2)

    # Euclidean distance
    euclidean_distance = np.linalg.norm(flat_a1 - flat_a2)

    # Correlation coefficient
    correlation_coefficient = np.corrcoef(flat_a1, flat_a2)[0, 1]

    # Angle between vectors
    cosine_angle = dot_product / (np.linalg.norm(flat_a1) * np.linalg.norm(flat_a2))
    angle = np.arccos(cosine_angle)

    return {
        'dot_product': dot_product,
        'euclidean_distance': euclidean_distance,
        'correlation_coefficient': correlation_coefficient,
        'angle_in_radians': angle,
        'angle_in_degrees': np.degrees(angle)
    }


# Assuming orthogonal_a1 and orthogonal_a2 are already defined
# interference_metrics = evaluate_interference(orthogonal_a1, orthogonal_a2)


def find_interference_locations(orthogonal_a1, orthogonal_a2):
    # Element-wise product or difference
    elementwise_product = np.multiply(orthogonal_a1, orthogonal_a2)
    elementwise_difference = np.subtract(orthogonal_a1, orthogonal_a2)

    # Find indices of max and min interference
    max_interference_idx_product = np.unravel_index(np.argmax(np.abs(elementwise_product)), elementwise_product.shape)
    min_interference_idx_product = np.unravel_index(np.argmin(np.abs(elementwise_product)), elementwise_product.shape)

    max_interference_idx_difference = np.unravel_index(
        np.argmax(np.abs(elementwise_difference)), elementwise_difference.shape
    )
    min_interference_idx_difference = np.unravel_index(
        np.argmin(np.abs(elementwise_difference)), elementwise_difference.shape
    )

    return {
        'max_interference_product': max_interference_idx_product,
        'min_interference_product': min_interference_idx_product,
        'max_interference_difference': max_interference_idx_difference,
        'min_interference_difference': min_interference_idx_difference
    }


# Assuming orthogonal_a1 and orthogonal_a2 are already defined
# interference_locations = find_interference_locations(orthogonal_a1, orthogonal_a2)


def compute_interference_metric(orthogonal_a1, orthogonal_a2):
    # Element-wise product
    return np.multiply(orthogonal_a1, orthogonal_a2)


def visualize_interference_distribution(interference_metric):
    # Flatten the interference metric to a 1D array
    flattened_interference = interference_metric.flatten()

    import matplotlib.pyplot as plt

    # Plotting the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_interference, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Interference Metric')
    plt.xlabel('Interference Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# Assuming orthogonal_a1 and orthogonal_a2 are already defined
# interference_metric = compute_interference_metric(orthogonal_a1, orthogonal_a2)
# visualize_interference_distribution(interference_metric)


def main():
    # Dummy weights for illustration (replace with actual model weights)
    weights_base = np.random.rand(10, 10)  # Base model weights
    weights_a1 = np.random.rand(10, 10)  # A1 model weights (specialized in code)
    weights_a2 = np.random.rand(10, 10)  # A2 model weights (specialized in summarization)

    # Combine the models
    new_model_weights = combine_models(weights_base, weights_a1, weights_a2)


if __name__ == '__main__':
    main()
