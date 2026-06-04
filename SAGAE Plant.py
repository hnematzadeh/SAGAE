# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 10:34:55 2026

@author: Kamyar
"""
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
# Define the model

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])




############## Model testing###############

# load the weights that yielded the best validation accuracy
# model.load_weights('plantResNet_TopIncludeFalse.weights.best.hdf5')
model.load_weights('D:/DataS Projects/DataS/LIME+/Plant disease/plantUpsampled4.weights.best.hdf5')

yhat = model.predict(X1_test)
y_pred = np.argmax(yhat, axis=1)
y_true = np.argmax(y1_test, axis=1)

# 3. Calculate Accuracy
accuracy = accuracy_score(y_true, y_pred)

# print(f"Total Samples Tested: {len(y_true)}")
# print(f"Correct Predictions: {np.sum(y_pred == y_true)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# 4. Optional: Detailed Breakdown per Plant Disease Class
# print("\nDetailed Analysis:")
# print(classification_report(y_true, y_pred))




######### Initial_population Generation  RANDOMLY  ##############
#################################################################

# 5 = Black Spot,  29 = Canker,  46, 48 = Greening,  55 = Healthy
X = X1_test[46].astype(np.float32)
Y = y1_test[46] 

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
import random

# 1. Configuration Parameters
N_SEGMENTS = 100
POP_SIZE = 10
orig_img = X.copy()

# 2. Generate the Uniform Master Superpixel Map (Once)
# This provides the structural skeleton for all your GA individuals
segments = slic(
    orig_img, 
    n_segments=N_SEGMENTS, 
    start_label=0, 
    compactness=10, 
    sigma=1, 
    channel_axis=-1
)
num_segments = np.max(segments) + 1
print(f"Generated Master Map with {num_segments} structural superpixels.")

initial_population = []

# 3. Superpixel-Level Initialization Loop
for i in range(POP_SIZE):
    # Create empty canvas containers matching the image shapes
    img_individual = np.zeros_like(orig_img)
    mask_individual = np.zeros((256, 256), dtype=np.float32)
    
    # Process at the superpixel level: 
    # Decide ON (1) or OFF (0) for each individual superpixel segment ID
    for seg_id in range(num_segments):
        # 50% chance a superpixel is turned ON or OFF
        is_on = random.choice([True, False])
        
        if is_on:
            # Find all pixel coordinates belonging to this specific superpixel
            seg_mask = (segments == seg_id)
            
            # Turn ON: Copy the whole superpixel block from the original image
            img_individual[seg_mask] = orig_img[seg_mask]
            mask_individual[seg_mask] = 1.0
        # Else remains OFF (black/0) by default
            
    # Store individual tracking its structural property
    initial_population.append({
        "image": img_individual.astype(np.float32),
        "mask": mask_individual,
        "n_segments": N_SEGMENTS
    })
    
    # 4. Visualize the generated solutions to verify they are block-based
    if i < 3:  # Plot the first 3 solutions
        plt.figure(figsize=(4, 4))
        plt.title(f"Initial Solution {i+1}")
        # Normalize display just in case values are scaled
        disp_img = img_individual / img_individual.max() if img_individual.max() > 0 else img_individual
        plt.imshow(mark_boundaries(disp_img, mask_individual.astype(int)))
        plt.axis('off')
        plt.show()
############################################################################
############################################################################



######################################################
#  5 = Black Spot,   29 = Canker,  46 = Greening,  55 = Healthy
X = X1_test[46].astype(np.float32)
Y = y1_test[46]  # 46 and 5 
############################
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
import random
segments_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
explainer = lime_image.LimeImageExplainer()

initial_population = []
# 4. The Generation Loop
for n_seg in segments_list:

    
    # explanation = explainer.explain_instance(X, model.predict, num_samples=10, segmentation_fn=slic)
    # Define the explanation with 50 segments
    explanation = explainer.explain_instance(
       X, 
       model.predict, 
       num_samples=1000, 
       # This tells SLIC to create exactly 50 superpixels
       segmentation_fn=lambda x: slic(x, n_segments=n_seg, start_label=0, compactness=10, sigma=0, channel_axis=-1)
       

     )
    image, mask = explanation.get_image_and_mask(model.predict(X.reshape(
    (1, 256, 256, 3))).argmax(axis=1)[0], negative_only=False, positive_only=True, hide_rest=True, num_features=2)


    initial_population.append({"image": image,"mask": mask, "n_segments": n_seg})
    plt.figure()
    plt.imshow(mark_boundaries((image), mask))
    plt.axis('off')
    plt.show()

# file_path = r'D:\DataS Projects\DataS\LIME+\Plant disease\initial population with default slic X1_test [5,29,46,55]\X1_test[46]\initial_population.pkl'
# with open('D:\DataS Projects\DataS\LIME+\Plant disease\initial population with default slic X1_test [5,29,46,55]\X1_test[46]\initial_population.pkl', 'rb') as f:
#     initial_population2 = pickle.load(f)
    
    

    
import pickle
# file_path = r'D:\DataS Projects\DataS\LIME+\Plant disease\X1_test[46]\initial_population.pkl'
file_path = r'D:\DataS Projects\DataS\LIME+\Plant disease\test [50]  top 2 features\initial_population.pkl'
# Save the list to a file
with open(file_path, 'wb') as f:
    pickle.dump(initial_population, f)

print("Population saved to initial_population.pkl")

# To load it back later:
# with open('D:\DataS Projects\DataS\LIME+\Plant disease\X1_test[46]\initial_population.pkl', 'rb') as f:
with open(r'D:\DataS Projects\DataS\LIME+\Plant disease\test [46]  top 2 features\initial_population.pkl', 'rb') as f:
    initial_population = pickle.load(f)
##############################
###########  GA FUNCTIONS ####

import random

def roulette_wheel_selection(population):
    fitness_values = [ind["fitness"] for ind in population]

    min_fit = min(fitness_values)
    if min_fit < 0:
        fitness_values = [f - min_fit for f in fitness_values]

    total = sum(fitness_values)
    if total == 0:
        return random.choice(population)

    probs = [f / total for f in fitness_values]
    idx = np.random.choice(len(population), p=probs)

    return population[idx]



# import numpy as np
import numpy as np
from skimage.transform import resize

def crossover(p1, p2, kernel_size=16): 
    """
    Performs block-based crossover on (256, 256, 3) images and (256, 256) masks.
    kernel_size: The size of the blocks to swap (default 16 for a 256px image).
    """
    img1, mask1 = p1["image"], p1["mask"]
    img2, mask2 = p2["image"], p2["mask"]
    
    h, w = mask1.shape  # Expected (256, 256)
    
    # 1. Create a small random grid for block crossover
    grid_h, grid_w = max(1, h // kernel_size), max(1, w // kernel_size)
    small_cross_map = np.random.rand(grid_h, grid_w) > 0.5
    
    # 2. Resize the grid back to the original image dimensions (256x256)
    # Using order=0 (nearest neighbor) to keep the blocks sharp
    cross_map = resize(small_cross_map, (h, w), order=0, 
                       preserve_range=True, anti_aliasing=False).astype(bool)
    
    # 3. Apply crossover to Masks (2D)
    child1_mask = np.where(cross_map, mask1, mask2)
    child2_mask = np.where(cross_map, mask2, mask1)
    
    # 4. Apply crossover to Images (3D)
    # np.expand_dims allows the 2D cross_map to broadcast across the 3 RGB channels
    cross_map_3d = np.expand_dims(cross_map, axis=-1)
    
    child1_img = np.where(cross_map_3d, img1, img2)
    child2_img = np.where(cross_map_3d, img2, img1)
    
    return (
        {"image": child1_img.astype(np.float32), "mask": child1_mask.astype(np.float32)},
        {"image": child2_img.astype(np.float32), "mask": child2_mask.astype(np.float32)}
    )



from scipy.ndimage import convolve
import numpy as np
from scipy.ndimage import convolve


# from tensorflow.keras.applications.resnet50 import preprocess_input

def calculate_fitness(individual, model, true_label):
    """
    Calculates individual fitness for the Leaf Disease Dataset (256x256x3).
    Maximizes model fidelity, maximizes sparsity, and enforces regional connectivity.
    """
    alpha = 1;  width = 256;  edge = 4
    global nfe
    nfe += 1
    
    img = individual["image"].copy()
    mask = individual["mask"]
    
    # 1. Ensure the image matches the 3-channel RGB ResNet input layout: (1, 256, 256, 3)
    # Handle any unexpected dimensions coming from the GA pipeline
    img_squeeze = img.squeeze()
    if img_squeeze.ndim == 2:
        # If grayscale, replicate across 3 channels
        img_rgb = np.stack([img_squeeze, img_squeeze, img_squeeze], axis=-1)
    else:
        img_rgb = img.reshape((256, 256, 3))
        
    model_input = np.expand_dims(img_rgb, axis=0).astype('float32')
    
    # # 2. Normalize and apply mandatory ResNet preprocessing
    # if model_input.max() > 1.0:
    #     model_input /= 255.0
    # model_input = preprocess_input(model_input)
    
    # 3. Fidelity Component
    predictions = model.predict(model_input, verbose=0)[0]
    fidelity = float(predictions[true_label])
    
    # 4. Sparsity Component (Using the 256x256 spatial footprint)
    mask_2d = mask.reshape((256, 256)).astype(float)
    coverage = np.mean(mask_2d) 
    if coverage == 0.0:
        
        return 0.0
    sparsity = 1.0 - coverage

    # 5. Connectivity / Smoothness Component (Scaled for 256x256)
    laplacian_kernel = np.array([[ 0,  1,  0],
                                 [ 1, -4,  1],
                                 [ 0,  1,  0]], dtype=float)
    
    laplacian = convolve(mask_2d, laplacian_kernel, mode='constant', cval=0.0)
    total_edges = np.sum(np.abs(laplacian))
    
    # LEAF SCALE FIX: A 256x256 mask boundary requires a larger baseline normalization cap
    # 1500.0 fits the typical perimeter length of localized leaf infection lesions
    max_expected_edges = alpha * width * edge  # 1500.0  
    connectivity_loss = total_edges / max_expected_edges
    connectivity = 1.0 - np.clip(connectivity_loss, 0.0, 1.0)

    # 6. Balanced Composite Weights
    w_fidelity = 0.6      # Increased slightly because leaf classification features are complex
    w_sparsity = 0.2      # Keeps the mask focused strictly on the infection site
    w_connectivity = 0.2  # Keeps the spot mask continuous instead of scattered
    
    fitness_score = (w_fidelity * fidelity) + (w_sparsity * sparsity) + (w_connectivity * connectivity)
    
    return float(fitness_score)



import numpy as np
import random
from skimage.segmentation import slic

def mutation(individual, original_image, mutation_rate=0.1):
    """
    Mutates an individual while strictly preserving its original superpixel architecture.
    """
    img = individual["image"].copy()  
    mask = individual["mask"].copy()  
    orig = original_image.astype(np.float32)
    
    # 1. READ the individual's inherent segment structure instead of hardcoding 100
    n_segments = individual.get("n_segments", 100) 
    
    # Regenerate the EXACT superpixel map this individual belongs to
    segments = slic(orig, n_segments=n_segments, start_label=0, compactness=10, sigma=1, channel_axis=-1)
    
    num_segments = np.max(segments) + 1
    num_mutations = max(1, int(mutation_rate * num_segments))
    
    # 2. Select random segments to flip
    chosen_segments = random.sample(range(num_segments), num_mutations)
    
    for seg_id in chosen_segments:
        seg_mask = (segments == seg_id)
        
        # STRICT FLIP: Base the choice on the absolute majority rule
        if np.mean(mask[seg_mask]) > 0.5:
            img[seg_mask] = 0        
            mask[seg_mask] = 0       
        else:
            img[seg_mask] = orig[seg_mask]
            mask[seg_mask] = 1
            
    return {
        "image": img.astype(np.float32), 
        "mask": mask.astype(np.float32),
        "n_segments": n_segments  # Keep tracking the structural property
    }
###################################################
###################################################
############################  GA PHASE ############
nfe = 0
target_class = np.argmax(Y)

avg_fitness_history = []

import numpy as np
import os
from skimage.segmentation import slic
# population = []

# for ind in initial_population:
#     # fitness = calculate_fitness(ind, model, target_class)
#     # ind["fitness"] = fitness
#     population.append(ind)

# nPop = len(population)



# --- Global GA Parameters ---
MAX_ITER = 100
pc = 0.9
pm = 0.4
mutation_mode = "normal"
stall = 0

best_score = -np.inf

# patience = int(0.2 * MAX_ITER)
patience = 30


best_fitness_history = []
avg_fitness_history = []
best_image_history = []

# --- Main GA Loop ---
# --- 1. Calculate Fitness for Iteration 0 ---
population = []
for ind in initial_population:
    score = calculate_fitness(ind, model, target_class)
    ind["fitness"] = score
    population.append(ind)
nPop = len(population)
nm = int(pm * nPop)
# --- 2. Sort Population ---
population.sort(key=lambda x: x["fitness"], reverse=True)

# --- 3. Store Results for Iteration 0 using .append() ---
best_score = population[0]["fitness"]
best_fitness_history.append(best_score)
best_image_history.append(population[0]["image"].copy())

current_scores = [ind["fitness"] for ind in population]
avg_fitness_history.append(np.mean(current_scores))

print(f"Iteration 0: Best Fitness = {best_fitness_history[0]:.4f}, NFE: {nfe}")



# --- 2. Initial Sort & History ---



for it in range(MAX_ITER):

    offspring = []

    # --- Crossover ---
    nc = int(pc * nPop)
    nc = nc if nc % 2 == 0 else nc - 1

    for _ in range(nc // 2):
        p1 = roulette_wheel_selection(population)
        p2 = roulette_wheel_selection(population)

        c1, c2 = crossover(p1, p2, 16)

        c1["fitness"] = calculate_fitness(c1, model, target_class)
        c2["fitness"] = calculate_fitness(c2, model, target_class)

        offspring.extend([c1, c2])

    # --- Mutation ---
    mutants = []
    nm = int(pm * nPop)

    for _ in range(nm):
        p = random.choice(population)
        m = mutation(p, X)

        m["fitness"] = calculate_fitness(m, model, target_class)
        mutants.append(m)

    # --- Survival ---
    population = population + offspring + mutants
    population = sorted(population, key=lambda x: x["fitness"], reverse=True)
    population = population[:nPop]
    
    avg_fitness = np.mean([ind["fitness"] for ind in population])
    avg_fitness_history.append(avg_fitness)
    # --- Tracking ---
    best = population[0]["fitness"]
    best_fitness_history.append(best)

    if best > best_score:
        best_score = best
        stall = 0
    else:
        stall += 1

    print(f"Iter {it}: Best = {best:.4f}, Stall={stall}, NFE={nfe}")

    if stall >= patience:
        print("Early stopping triggered")
        break
    
    
    
best_solution = population[0]



# np.save(r"C:\Users\User\Desktop\IMAGES\Lime guided\greening random.npy", best_fitness_history)

# np.save("D:\DataS Projects\DataS\LIME+\Plant disease\Figures in the paper\SAGA 3 Plant\SAGA with random initialization\ Healthy LIME.npy", best_fitness_history)
######## model predict for best image

import numpy as np

# 1. Identify the best individual (assumes population is sorted by fitness)
best_solution = population[0]

best_img = best_solution["image"]   # Should be (256, 256, 3)
best_mask = best_solution["mask"]   # Should be (256, 256)

# 2. Reshape for Model input (1, 256, 256, 3)
# We use the RGB image directly for the leaf model
input_tensor = best_img.reshape((1, 256, 256, 3)).astype('float32')

# 3. Get prediction
prediction = model(input_tensor, training=False).numpy()

# 4. Display Results
print("-" * 30)
print(f"Leaf Disease Explanation Results")
print("-" * 30)
print(f"Target Class: {target_class}")
print(f"Fidelity: {prediction[0][target_class]:.2f}")
print(f"Sparsity: {(1 - np.mean(best_mask)):.2f} ")
print("-" * 30)


############## fitness history plot###################

import matplotlib.pyplot as plt
import numpy as np

# 1. Determine the actual number of iterations completed 
# (In case the GA stopped early)
min_length = min(len(best_fitness_history), len(avg_fitness_history))

# 2. Slice both arrays and the X-axis to this length
iterations = np.arange(min_length)
plot_best = best_fitness_history[:min_length]
plot_avg = avg_fitness_history[:min_length]

# Use a clean style for better visualization
plt.style.use('seaborn-v0_8-muted') 
plt.figure(figsize=(14, 9))

# 3. Plotting with bold lines for clarity
plt.plot(iterations, plot_best, label='Best Fitness (Top Explanation)', 
         color='#2ecc71', linewidth=9)
plt.plot(iterations, plot_avg, label='Average Fitness (Population)',
         color='red', linewidth=4.5, linestyle='--')

# 4. Dynamic Y-Axis
# Leaf datasets often have lower initial fitness than MNIST. 
# Autoscale ensures you actually see the progress curve.
plt.autoscale(enable=True, axis='y')

# 5. Formatting for high-resolution leaf results
plt.title('Greening', fontsize=48, pad=25)
plt.xlabel('Iteration', fontsize=40)
plt.ylabel('Fitness Score', fontsize=40)

plt.tick_params(axis='both', which='major', labelsize=34)

plt.legend(fontsize=18, loc='best', frameon=True, shadow=True)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout() 

plt.show()
########## BLUE BACKGROUND GENERATION - GA generated ###################

import matplotlib.pyplot as plt
import numpy as np

# 1. Extract the actual numpy arrays
vis_rgb = best_solution["image"].copy()
mask_2d = np.squeeze(best_solution["mask"])

# 2. Normalize to 0-255 uint8 range for plotting
if vis_rgb.max() <= 1.0:
    vis_rgb = (vis_rgb * 255).astype(np.uint8)
else:
    vis_rgb = vis_rgb.astype(np.uint8)

# 3. Highlight inactive areas with Light Sky Blue
# RGB for Light Sky Blue: R=135, G=206, B=250
light_sky_blue = [135, 206, 250] 
inactive_pixels = (mask_2d == 0)
vis_rgb[inactive_pixels] = light_sky_blue

# 4. Plotting
plt.figure(figsize=(10, 10))
plt.imshow(vis_rgb)

# Update title to reflect the specific color
# plt.title("SAGA Explanation ", fontsize=32, pad=20)

# Remove axis lines and numbers
plt.axis('off')

plt.tight_layout()
plt.show()
###################################################


import numpy as np

def FS(individual, model, target_idx):
    img = individual["image"]
    mask = individual["mask"]

    # 1. Prepare image for model (Batch, H, W, C)
    temp_final = np.expand_dims(img, axis=0)
    
    # Ensure float32 and 0-1 scaling for ResNet
    temp_final = temp_final.astype('float32')
    if temp_final.max() > 1.0:
        temp_final /= 255.0

    # 2. Get the probability for the specific target class
    # model.predict returns something like [[0.1, 0.9, 0.0, 0.0]]
    prediction = model.predict(temp_final, verbose=0) 
    
    # Get the value at row 0, column target_idx
    fidelity = prediction[0][target_idx]

    # 3. Calculate sparsity (how much of the image is hidden)
    sparsity = 1.0 - np.mean(mask)
    # binary_mask = (mask > 0).astype(float)
    # sparsity = 1.0 - np.mean(binary_mask)

    return fidelity, sparsity

# --- HOW TO CALL IT CORRECTLY ---
# binary_mask = np.any(best_solution != 0, axis=-1).astype(int)

# Convert Y to a single integer once
target_idx = np.argmax(Y) 

# Pass that integer into the function
f, s = FS(best_solution, model, target_idx)
# f, s = FS(refined_solution_dict, model, target_idx)


print('--- GA Solution ---')
print(f'fidelity: {f:.2f}')
print(f'sparsity: {s:.2f}')


########### LIME fidelity and sparsity

def FS(individual, model, target_class):
    img = individual["image"]
    mask = individual["mask"]
    
    # 1. Add the batch dimension: converts (256, 256, 3) to (1, 256, 256, 3)
    temp_final = img[np.newaxis, ...].astype('float32')
    
    # 2. FIX: Pass the batched 'temp_final' to the model instead of 'img'
    prediction = model(temp_final, training=False).numpy()
    
    # Extract prediction probability for the target class
    fidelity = prediction[0][target_class]
    
    # Sparsity (1 - mean of the binary mask)
    sparsity = 1 - np.mean(mask)
    
    return fidelity, sparsity


lime_solution = {
    "image": initial_population[2]['image'],
    "mask": initial_population[2]['mask']
}

f_lime, s_lime = FS(lime_solution, model, int(np.argmax(Y)))
# fitness_lime = 0.8 * f_lime + 0.2 * s_lime

print('\n--- LIME Solution ---')
print('fidelity:', np.round(f_lime, 2))
print('sparsity:', np.round(s_lime, 2))
# print('fitness :', np.round(fitness_lime, 2))
plt.imshow(initial_population[2]['image'])

##################################################

explanations = []

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

# Assuming X1_test[5] is already shape (256, 256, 3)
X = X1_test[5]
Y = y1_test[5]

explainer = lime_image.LimeImageExplainer()



img_rgb = X.astype('double')

initial_population = [] 
explanations = [] # Ensure this list is initialized before the loop
segments_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for n_seg in segments_list:
    # Set channel_axis=-1 because your image data is channels-last (256, 256, 3)
    segmenter = SegmentationAlgorithm(
        'slic', 
        n_segments=n_seg, 
        start_label=0, 
        compactness=10, 
        sigma=1, 
        channel_axis=-1
    )

    explanation = explainer.explain_instance(
        img_rgb,
        model.predict,
        segmentation_fn=segmenter,
        num_samples=1000,
        hide_color=None  # Keeps background black
    )

    # Extracted top mask and features
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=2, 
        hide_rest=True
    )

    # Ensure mask is 2D binary
    mask = (mask > 0).astype(np.float32)

    # Normalize if LIME scales to 0-255 range
    if temp.max() > 1.0:
        temp = temp / 255.0

    initial_population.append({
        "image": temp.astype('float32'),
        "mask": mask
    })

# --- CRITICAL FIX: Moved these OUTSIDE the segment generation loop ---

# 1. Populate your explanations list once the loop is fully complete
for item in initial_population:
    explanations.append(item['mask'])
    plt.figure()
    plt.imshow(item['image'])
    plt.show()























explanations = []
###########################################
################  CONSISTENCY ANALYSIS ######


explanations.append(best_solution["mask"])

import numpy as np

def calculate_jaccard(mask1, mask2):
    """Calculates the Jaccard Index (Intersection over Union) for two binary masks."""
    # Ensure masks are boolean/binary
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 1.0  # Both masks are empty, technically identical
    return intersection / union

def evaluate_stability(image, model, label, n_runs=3):
    """
    Performs multiple independent runs and calculates the average 
    pairwise Jaccard similarity (Stability).
    """
    # explanations = []
    
    print(f"Starting Stability Analysis for {n_runs} runs...")
    
    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs} in progress...")
        # Replace this with your actual combined GA + HC call
        # result = run_ga_hc_pipeline(image, model, label)
        # explanations.append(result['mask'])
        
        # Placeholder for demonstration:
        # explanations.append(refined_solution['mask']) 
        pass

    # Calculate pairwise Jaccard similarities
    jaccard_scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            score = calculate_jaccard(explanations[i], explanations[j])
            jaccard_scores.append(score)
            print(f"Jaccard(E_{i+1}, E_{j+1}) = {score:.2f}")

    # Final Stability Score
    stability = np.mean(jaccard_scores)
    print(f"\nFinal Stability Score: {stability:.2f}")
    
    return stability

# Example Usage:
stability_score = evaluate_stability(X, model, Y, n_runs=3)







########################## PLOT
####################################
####################################
################### Fidelity Sparsity Trade off Canker



import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  

# 1. Prepare data for LIME (using your exact variable definitions)
fidelity_lime = [0.01, 0.02, 0.01, 0.47, 0.12, 0.08, 0.05, 0.08, 0.09, 0.13]
sparsity_lime = [0.83, 0.83, 0.86, 0.93, 0.92, 0.92, 0.95, 0.96, 0.97, 0.95]

# 2. Prepare data for SAGA
fidelity_saga = 0.97
sparsity_saga = 0.94

# 3. Initialize the plot
fig, ax = plt.subplots(figsize=(10, 7))

# 4. Plot LIME entries with increased circle size (s=1200)
ax.scatter(
    sparsity_lime, 
    fidelity_lime, 
    color='salmon',         
    edgecolors='black',     
    linewidth=1.8,          
    s=1200,                 # <-- Significantly increased size
    alpha=0.75,              
    label='LIME',           # <-- Simplified legend name
    zorder=5
)

# 5. Plot SAGA with increased circle size (s=1200)
ax.scatter(
    sparsity_saga, 
    fidelity_saga, 
    color='lightblue',      
    edgecolors='black',     
    linewidth=1.8,
    s=1200,                 # <-- Significantly increased size
    label='SAGAE',           # <-- Simplified legend name
    zorder=7
)

# 6. Configure labels, title, gridlines, and legends
ax.set_xlabel('Sparsity', fontsize=28, labelpad=8)
ax.set_ylabel('Fidelity', fontsize=28, labelpad=8)
ax.set_title('Fidelity vs Sparsity Trade-off: Canker', fontsize=32, pad=12)

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
ax.tick_params(axis='both', which='major', labelsize=18)

# Natural axis scaling tightly wrapping your data limits
all_sparsity = sparsity_lime + [sparsity_saga]
all_fidelity = fidelity_lime + [fidelity_saga]
ax.set_xlim(min(all_sparsity) - 0.05, max(all_sparsity) + 0.05)
ax.set_ylim(min(all_fidelity) - 0.05, max(all_fidelity) + 0.05)
ax.tick_params(axis='both', which='major', labelsize=24)

ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

# Simplified clean legend with larger marker scales
ax.legend(
    loc='upper right', 
    fontsize=16, 
    markerscale=0.6,       
    handletextpad=0.8,
    frameon=True
)

plt.tight_layout()
plt.show()




########## Fidelity Sparsity Trade off Black Spot

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  

# 1. Prepare data for LIME (Updated for Black spot)
fidelity_lime = [1, 1, 1, 1, 1, 0.97, 1, 0.88, 0.98, 0.85]
sparsity_lime = [0.77, 0.86, 0.85, 0.93, 0.93, 0.94, 0.95, 0.96, 0.96, 0.96]

# 2. Prepare data for SAGA (Updated for Black spot)
fidelity_saga = 1
sparsity_saga = 0.95

# 3. Initialize the plot
fig, ax = plt.subplots(figsize=(10, 7))

# 4. Plot LIME entries with high opacity, black borders, and large circle size (s=1200)
ax.scatter(
    sparsity_lime, 
    fidelity_lime, 
    color='salmon',         
    edgecolors='black',     
    linewidth=1.8,          
    s=1200,                 
    alpha=0.75,              
    label='LIME',           
    zorder=5
)

# 5. Plot SAGA with large circle size (s=1200)
ax.scatter(
    sparsity_saga, 
    fidelity_saga, 
    color='lightblue',      
    edgecolors='black',     
    linewidth=1.8,
    s=1200,                 
    label='SAGAE',           
    zorder=7
)

# 6. Configure labels, title, gridlines, and legends
ax.set_xlabel('Sparsity', fontsize=28, labelpad=8)
ax.set_ylabel('Fidelity', fontsize=28, labelpad=8)
ax.set_title('Fidelity vs Sparsity Trade-off: Black spot', fontsize=32, pad=12) # <-- Updated Title

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) # Ensures uniform floating points on Y-axis too

# Natural axis scaling adjusted dynamically for the tight clustering of Black spot data
all_sparsity = sparsity_lime + [sparsity_saga]
all_fidelity = fidelity_lime + [fidelity_saga]
ax.set_xlim(min(all_sparsity) - 0.04, max(all_sparsity) + 0.04)
ax.set_ylim(min(all_fidelity) - 0.04, max(all_fidelity) + 0.04)
ax.tick_params(axis='both', which='major', labelsize=24)

ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

# Simplified clean legend
ax.legend(
    loc='lower left', 
    fontsize=16, 
    markerscale=0.6,       
    handletextpad=0.8,
    frameon=True
)

plt.tight_layout()
plt.show()


######################## Fidelity Sparsity Trade off Greening

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  

# 1. Prepare data for LIME (Updated Greening Data)
sparsity_lime = [0.63, 0.87, 0.88, 0.92, 0.93, 0.95, 0.96, 0.97, 0.97, 0.97]
fidelity_lime = [0.19, 0.11, 0.29, 0.36, 0.69, 0.86, 0.71, 0.86, 0.82, 0.87]

# 2. Prepare data for SAGA - GREENING (Updated)
sparsity_saga = 0.98
fidelity_saga = 0.94

# 3. Initialize the plot
fig, ax = plt.subplots(figsize=(10, 7))

# 4. Plot LIME entries with high opacity, black borders, and large circle size (s=1200)
ax.scatter(
    sparsity_lime, 
    fidelity_lime, 
    color='salmon',         
    edgecolors='black',     
    linewidth=1.8,          
    s=1200,                 
    alpha=0.75,              
    label='LIME',           
    zorder=5
)

# 5. Plot SAGA with large circle size (s=1200)
ax.scatter(
    sparsity_saga, 
    fidelity_saga, 
    color='lightblue',      
    edgecolors='black',     
    linewidth=1.8,
    s=1200,                 
    label='SAGAE',           
    zorder=7
)

# 6. Configure labels, title, gridlines, and legends
ax.set_xlabel('Sparsity', fontsize=28, labelpad=8)
ax.set_ylabel('Fidelity', fontsize=28, labelpad=8)
ax.set_title('Fidelity vs Sparsity Trade-off: Greening', fontsize=32, pad=12)

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 

# Natural axis scaling adjusted dynamically for the new Greening distribution
all_sparsity = sparsity_lime + [sparsity_saga]
all_fidelity = fidelity_lime + [fidelity_saga]
ax.set_xlim(min(all_sparsity) - 0.05, max(all_sparsity) + 0.05)
ax.set_ylim(min(all_fidelity) - 0.05, max(all_fidelity) + 0.05)
ax.tick_params(axis='both', which='major', labelsize=24)

ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

# Simplified clean legend placed out of the way of the data cluster
ax.legend(
    loc='upper left', 
    fontsize=16, 
    markerscale=0.6,       
    handletextpad=0.8,
    frameon=True
)

plt.tight_layout()
plt.show()
#################Fidelity Sparsity Healthy Trade Off########################

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  

# 1. Prepare data for LIME (Updated for Healthy)
fidelity_lime = [0.45, 0.07, 0.07, 0.02, 0.02, 0.01, 0.03, 0.02, 0.04, 0.01]
sparsity_lime = [0.66, 0.84, 0.85, 0.92, 0.93, 0.95, 0.96, 0.97, 0.97, 0.98]

# 2. Prepare data for SAGA (Updated for Healthy)
fidelity_saga = 0.99
sparsity_saga = 0.35

# 3. Initialize the plot
fig, ax = plt.subplots(figsize=(10, 7))

# 4. Plot LIME entries with high opacity, black borders, and large circle size (s=1200)
ax.scatter(
    sparsity_lime, 
    fidelity_lime, 
    color='salmon',         
    edgecolors='black',     
    linewidth=1.8,          
    s=1200,                 
    alpha=0.75,              
    label='LIME',           
    zorder=5
)

# 5. Plot SAGA with large circle size (s=1200)
ax.scatter(
    sparsity_saga, 
    fidelity_saga, 
    color='lightblue',      
    edgecolors='black',     
    linewidth=1.8,
    s=1200,                 
    label='SAGAE',           
    zorder=7
)

# 6. Configure labels, title, gridlines, and legends
ax.set_xlabel('Sparsity', fontsize=28, labelpad=8)
ax.set_ylabel('Fidelity', fontsize=28, labelpad=8)
ax.set_title('Fidelity vs Sparsity Trade-off: Healthy', fontsize=32, pad=12) # <-- Updated Title

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 

# Natural axis scaling adjusted dynamically for the Healthy data layout
all_sparsity = sparsity_lime + [sparsity_saga]
all_fidelity = fidelity_lime + [fidelity_saga]
ax.set_xlim(min(all_sparsity) - 0.05, max(all_sparsity) + 0.05)
ax.set_ylim(min(all_fidelity) - 0.05, max(all_fidelity) + 0.05)
ax.tick_params(axis='both', which='major', labelsize=24)

ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

# Simplified clean legend
ax.legend(
    loc='upper right',       # Moved to upper right since SAGA sits on the upper left for this dataset
    fontsize=16, 
    markerscale=0.6,       
    handletextpad=0.8,
    frameon=True
)

plt.tight_layout()
plt.show()


###################################################################


 
























import matplotlib.pyplot as plt
import numpy as np

# --- 1. Simulation Data Setup ---
# (Replace these dummy arrays with your actual saved 'best_fitness_history' lists)
# Assuming SAGA + LIME converges faster due to guided initialization
lime_init_history = Digit8lime
random_init_history = Digit8random

# --- 2. Plotting Configuration ---
plt.figure(figsize=(10, 6))

# Plot LIME curve across its 59 generations
plt.plot(
    range(len(lime_init_history)), 
    lime_init_history, 
    label=f"SAGA (LIME Initialization)", 
    color="#1f77b4", 
    linewidth=2.5
)

# Plot Random curve across its 81 generations
plt.plot(
    range(len(random_init_history)), 
    random_init_history, 
    label=f"SAGA (Random Initialization)", 
    color="#ff7f0e", 
    linewidth=2.5
)

# --- 3. Labels and Style Formatting ---
plt.xlabel("Iteration", fontsize=28, labelpad=10)
plt.ylabel("Best Fitness Score", fontsize=28, labelpad=10)
plt.title("Digit 8", fontsize=32, pad=15)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.ylim(0.71,0.795)

# Keep the view scaled snugly around the length of the longer run
max_iters = max(len(lime_init_history), len(random_init_history))
plt.xlim(-1, max_iters + 1)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="right", fontsize=12, frameon=True, shadow=True)

plt.tight_layout()
plt.show()







plt.imshow(X1_test[49])
plt.axis('off')
y1_test[49]


import matplotlib.pyplot as plt
import numpy as np

# 1. Get the image and mask from your dictionary
img = initial_population[8]['image']
mask = initial_population[8]['mask']

# 2. Make a copy so we don't alter the original data
modified_img = img.copy()

# 3. Where the mask is False (background), change pixel values to Yellow [R, G, B]
# (Use 1.0, 1.0, 0.0 if your image uses floats between 0 and 1)
light_peach = [0.996, 0.922, 0.839]
modified_img[mask == 0] = light_peach

# 4. Display the result
plt.imshow(modified_img)
plt.axis('off')
plt.show()
