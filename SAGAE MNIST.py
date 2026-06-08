ng# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:10:14 2026

@author: Kamyar
"""

################### MNIST Model Genearion    #############################
from tensorflow import keras
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test) = mnist.load_data()
# x_train.shape
# (60000, 28, 28) the images are black and white
# plt.imshow(x_train[10], cmap='gray')



#### Preprocessing
## we normalize images
x_train = x_train.reshape((-1,28,28,1)).astype('float32')/255
x_test = x_test.reshape((-1,28,28,1)).astype('float32')/255

### model definition

model = keras.Sequential()
model.add(keras.layers.Conv2D(64, kernel_size = 3*3, strides = (1,1), padding = 'valid', activation= 'relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))


# ### model compile
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ### model fit
# early_stop= keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights=True,)
# history=model.fit(x_train,y_train, validation_split = 0.1, batch_size=256, epochs = 50, callbacks = [early_stop])


# model.save_weights('D:/DataS Projects/DataS/LIME+/MNIST/mnist.weights.h5')
model.load_weights('D:/DataS Projects/DataS/LIME+/MNIST/mnist.weights.h5')



y_pred = model.predict(x_test)

# Get the index of the highest probability for each prediction
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
############################################################
####################  MElanoma Model Generation ############
# Class name to the index
#class_2_indices = train_generator.class_indices

#############  Initial population generation Randomly #############


######### MNIST Random Superpixel-Level Initialization ##############
#####  5=15, 4=19, 3=30, 8=110

# --- Select MNIST Target Sample ---
X = x_test[19]        # Grayscale image shape, e.g., (28, 28, 1) or (28, 28)
Y = y_test[19]
target_idx = Y

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
import random

# 1. Configuration Parameters
N_SEGMENTS = 100
POP_SIZE = 10

# Ensure image shape is formatted correctly for standard SLIC processing
orig_img = X.squeeze()  # Shape becomes exactly (28, 28)

# Replicate to RGB format temporarily so shapes become (28, 28, 3)
# img_rgb = np.repeat(orig_img[:, :, np.newaxis], 3, axis=-1).astype('double')

# 2. Generate the Uniform Master Superpixel Map (Once)
segments = slic(
    orig_img, 
    n_segments=N_SEGMENTS, 
    start_label=0, 
    compactness=1,  
    sigma=0, 
    channel_axis=None  # <-- FIXED: Points to the trailing channel axis (index 2)
)
num_segments = np.max(segments) + 1
print(f"Generated MNIST Master Map with {num_segments} structural superpixels.")

initial_population = []

# 3. Superpixel-Level Selection Loop
for i in range(POP_SIZE):
    img_individual = np.zeros((28, 28, 3), dtype=np.float32)
    mask_individual = np.zeros((28, 28), dtype=np.float32)
    
    # Decide ON or OFF for each unique segment block
    for seg_id in range(num_segments):
        is_on = random.choice([True, False])
        
        if is_on:
            seg_mask = (segments == seg_id) # 2D boolean map matching (28, 28)
            
            # Replicate the grayscale feature value across all 3 channels
            for ch in range(3):
                img_individual[seg_mask, ch] = orig_img[seg_mask]
            
            mask_individual[seg_mask] = 1.0
    
    # Normalize values to ensure they stay strictly within the [0, 1] range
    if img_individual.max() > 1.0:
        img_individual /= 255.0
    
    # Store structural solution tracking components
    initial_population.append({
        "image": img_individual.astype('float32'),
        "mask": mask_individual,
        "n_segments": N_SEGMENTS
    })
    
    # 4. Optional: Plot a few solutions to verify block distributions
    if i < 3:
        plt.figure(figsize=(3, 3))
        plt.title(f"Initial Ind {i+1}")
        plt.imshow(mark_boundaries(img_individual, mask_individual.astype(int)))
        plt.axis('off')
        plt.show()

###########################################################################

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries


#####  5=15, 4=19, 3=30, 8=110
X = x_test[110]
Y = y_test[110]
target_idx = Y




explainer = lime_image.LimeImageExplainer()

# def predict_wrapper(images):
#     gray_images = images[:, :, :, 0:1]
#     return model(gray_images, training=False).numpy()


def predict_wrapper(images):
    # LIME generates RGB-like images, we take only 1 channel and resize to 28x28
    # Then we normalize (though your model was trained on 0-1, so we ensure that here)
    gray_images = images[:, :, :, 0:1] 
    return model.predict(gray_images).astype('float32')


img_rgb = np.repeat(X, 3, axis=-1).astype('double')

initial_population = [] 
segments_list = [10,20,30,40,50,60,70,80,90,100]
                                                                                                                  
for n_seg in segments_list:
    # Use a fixed segmenter for this iteration
    segmenter = SegmentationAlgorithm('slic', n_segments=n_seg, start_label=0, sigma=0)

    explanation = explainer.explain_instance(
        img_rgb,
        predict_wrapper,
        segmentation_fn=segmenter,
        num_samples=1000,
        hide_color=0  # <--- CRITICAL: Keeps background black/consistent
    )

    # num_features=1 might be too low for 100 segments; 
    # consider num_features=int(n_seg * 0.1) for better initial variety
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        negative_only=False,
        num_features=1, 
        hide_rest=True
    )

    # ensure mask is 2D binary
    mask = (mask > 0).astype(np.float32)

    # --- ADDED: Normalization Check ---
    # LIME sometimes returns images in 0-255 range if input was double.
    # We ensure it's 0-1 for your MNIST model.
    if temp.max() > 1.0:
        temp = temp / 255.0

    initial_population.append({
        "image": temp.astype('float32'),
        "mask": mask,
        "n_segments": n_seg
    })
    plt.figure()
    plt.imshow(mark_boundaries(temp, mask.astype(int)))
    plt.axis('off')
    plt.show()


import pickle
# file_path = r'D:\DataS Projects\DataS\LIME+\Plant disease\X1_test[46]\initial_population.pkl'
file_path = r'D:\DataS Projects\DataS\LIME+\MNIST\Digit 8 top 1 feature\initial_population.pkl'
# Save the list to a file
with open(file_path, 'wb') as f:
    pickle.dump(initial_population, f)


# To load it back later:
# with open('D:\DataS Projects\DataS\LIME+\Plant disease\X1_test[46]\initial_population.pkl', 'rb') as f:
with open(r'D:\DataS Projects\DataS\LIME+\MNIST\Digit 8 top 1 feature\initial_population.pkl', 'rb') as f:
    initial_population = pickle.load(f)

############# GA operators
import numpy as np
from scipy.ndimage import convolve

def calculate_fitness(individual, model, true_label):
    """
    Calculates individual fitness by maximizing model classification prediction 
    while minimizing a Laplacian edge penalty to encourage smooth, continuous masks.
    """
    alpha = 1;  width = 28;  edge = 4

    global nfe
    nfe += 1
    # 1. Extract and normalize structural properties
    img = individual["image"]
    mask = individual["mask"]
    
    # 2. Reshape image to match model input expectations: (1, 28, 28, 1)
    # LIME images may contain 3 channels; we extract only the first channel for MNIST
    if img.ndim == 3 and img.shape[-1] == 3:
        model_input = img[:, :, 0:1]
    else:
        model_input = img.reshape((28, 28, 1))
        
    model_input = np.expand_dims(model_input, axis=0).astype('float32')
    
    # 3. Model Prediction Component
    predictions = model.predict(model_input, verbose=0)[0]
    fidelity = predictions[true_label]
    
    # 4. Smoothness / Continuity Penalty (Laplacian Edge Check)
    # Standard 3x3 Laplacian Kernel to find sharp transitions in mask boundaries
    laplacian_kernel = np.array([[ 0,  1,  0],
                                 [ 1, -4,  1],
                                 [ 0,  1,  0]], dtype=float)
    
    # CRITICAL FIX: Squeeze both arrays to be strictly 2D (H, W) to eliminate SciPy shape errors
    mask_2d = mask.squeeze().astype(float)
    # kernel_2d = laplacian_kernel.squeeze()
    
    # Compute edge violations
    laplacian = convolve(mask_2d, laplacian_kernel, mode='constant')
    active_pixels = np.sum(mask)
    total_edges = np.sum(np.abs(laplacian))
    max_expected_edges = alpha * width * edge  # 1500.0  
    connectivity_loss = total_edges / max_expected_edges
    connectivity = 1.0 - np.clip(connectivity_loss, 0, 1)

    # --- 3. Sparsity (Size Constraint) ---
    # Rewards smaller masks to find the specific site of infection
    coverage = np.mean(mask) 
    if coverage == 0.0:
        
        return 0.0
    sparsity = 1.0 - coverage
    fitness_score = (0.6 * fidelity) + (0.2 * sparsity) + (0.2 * connectivity)

    
    return float(fitness_score)

import numpy as np
from skimage.transform import resize

def crossover(p1, p2, kernel_size=2): 
    # Use .squeeze() to turn (28,28,1) or (28,28,3) into (28,28) if possible
    # If it is (28,28,3), we take the mean to force it to (28,28)
    def to_2d(img):
        img = np.array(img)
        if img.ndim == 3:
            if img.shape[-1] == 3:
                return np.mean(img, axis=-1)
            else:
                return img.squeeze(-1)
        return img

    img1, mask1 = to_2d(p1["image"]), to_2d(p1["mask"])
    img2, mask2 = to_2d(p2["image"]), to_2d(p2["mask"])
    
    h, w = mask1.shape
    grid_h, grid_w = max(1, h // kernel_size), max(1, w // kernel_size)
    small_cross_map = np.random.rand(grid_h, grid_w) > 0.5
    
    # Resize map to match 28x28
    cross_map = resize(small_cross_map, (h, w), order=0, 
                       preserve_range=True, anti_aliasing=False).astype(bool)
    

    # Apply crossover on 2D arrays
    child1_img = np.where(cross_map, img1, img2)
    child2_img = np.where(cross_map, img2, img1)
    
    child1_mask = np.where(cross_map, mask1, mask2)
    child2_mask = np.where(cross_map, mask2, mask1)
    
    return (
        {"image": child1_img.astype(np.float32), "mask": child1_mask.astype(np.float32)},
        {"image": child2_img.astype(np.float32), "mask": child2_mask.astype(np.float32)}
    )


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






import numpy as np
import random
from skimage.segmentation import slic

def mutation(individual, original_image, mutation_rate=0.2):
    """
    Mutates an MNIST individual while strictly preserving its original 
    superpixel architecture and filtering out dimensional channel noise.
    """
    # 1. Force original image to a clean 2D slice strictly for SLIC to trace coordinates
    slic_input = original_image.reshape((28, 28)).astype(np.float32)
    
    # 2. Extract a clean, single-channel 2D mask (28x28) to fix the 21952 size bug
    raw_mask = individual["mask"]
    if raw_mask.size > 784:
        # If the size is expanded (like 21952), extract just the first 2D slice
        mask_2d = raw_mask.reshape((28, 28, -1))[:, :, 0].copy()
    else:
        mask_2d = raw_mask.reshape((28, 28)).copy()
        
    # Extract image and ensure it matches the 2D spatial footprint
    img_2d = individual["image"].reshape((28, 28, -1))[:, :, 0].copy()
    n_segments = individual.get("n_segments", 50) 
    
    # 3. Regenerate the exact superpixel map layout (shape: 28, 28)
    segments = slic(
        slic_input, 
        n_segments=n_segments, 
        compactness=10, 
        sigma=0,              
        start_label=0, 
        channel_axis=None     
    )
    
    num_segments = np.max(segments) + 1
    num_mutations = max(1, int(mutation_rate * num_segments))
    chosen_segments = random.sample(range(num_segments), num_mutations)
    
    # 4. Mutate clean 2D matrices directly (no broadcasting traps)
    for seg_id in chosen_segments:
        seg_mask_2d = (segments == seg_id)
        
        # Check the spatial mask region to decide whether to flip
        if np.mean(mask_2d[seg_mask_2d]) > 0.5:
            img_2d[seg_mask_2d] = 0
            mask_2d[seg_mask_2d] = 0
        else:
            img_2d[seg_mask_2d] = slic_input[seg_mask_2d]
            mask_2d[seg_mask_2d] = 1
            
    return {
        "image": img_2d.reshape((28, 28, 1)).astype(np.float32), 
        "mask": mask_2d.reshape((28, 28, 1)).astype(np.int32), # int32 protects mark_boundaries
        "n_segments": n_segments                 
    }
############################  SAGAE ##################
import math
nfe = 0

avg_fitness_history = []

import numpy as np
import os
from skimage.segmentation import slic


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
    score = calculate_fitness(ind, model, Y)
    ind["fitness"] = score
    population.append(ind)
    
nPop = len(population)

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

        c1, c2 = crossover(p1, p2, 2)

        c1["fitness"] = calculate_fitness(c1, model, Y)
        c2["fitness"] = calculate_fitness(c2, model, Y)

        offspring.extend([c1, c2])

    # --- Mutation ---
    mutants = []
    nm = math.ceil(pm * nPop)

    for _ in range(nm):
        p = random.choice(population)
        m = mutation(p, X)

        m["fitness"] = calculate_fitness(m, model, Y)
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

import matplotlib.pyplot as plt

plt.imshow(best_solution["image"])
plt.title("SAGA Explanation")
plt.axis('off')
plt.show()

# np.save(r"C:\Users\User\Desktop\IMAGES\Lime guided\Digit 8 random.npy", best_fitness_history)


# np.save("D:\DataS Projects\DataS\LIME+\Plant disease\Figures in the paper\SAGA 3 Plant\SAGA with random initialization\ Digit 8 Random.npy", best_fitness_history)



######## model predict for best image

best_solution = population[0]

best_img = best_solution["image"]
best_mask = best_solution["mask"]

# 1. Handle shape: Ensure it is 2D (28, 28) by removing extra dimensions
if best_img.ndim == 3:
    grayscale_img = best_img.squeeze()
else:
    grayscale_img = best_img

# 2. Reshape for Model input (Batch, Height, Width, Channel)
input_image = grayscale_img.reshape((1, 28, 28, 1)).astype('float32')

# 3. Get prediction
prediction = model(input_image, training=False).numpy()

print(f"Prediction for target class {Y}: {prediction[0][Y]:.4f}")

# prediction = model.predict(input_image, verbose=0)[0]
# fidelity = prediction[target_idx]
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
plt.title('Digit 8', fontsize=48, pad=25)
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

# Use a fresh copy to avoid modifying the original solution
best_img = best_solution["image"].copy()
best_mask = best_solution["mask"].copy()

# 1. FORCE 2D: Ensure both are exactly (28, 28)
# Squeeze handles (1, 28, 28), (28, 28, 1), or (1, 28, 28, 1)
img_2d = np.squeeze(best_img)
mask_2d = np.squeeze(best_mask)

# If the image is still 3D (e.g., RGB), force it to grayscale
if img_2d.ndim == 3:
    img_2d = np.mean(img_2d, axis=-1)

# 2. Normalize to 0–255
if img_2d.max() <= 1.0:
    img_disp = (img_2d * 255).astype(np.uint8)
else:
    img_disp = img_2d.astype(np.uint8)

# 3. Create RGB image
# By stacking three (28, 28) arrays, we get exactly (28, 28, 3)
vis_rgb = np.dstack([img_disp, img_disp, img_disp])

# 4. Apply the blue mask
# Identify pixels where the mask is 0
inactive_pixels = (mask_2d == 0)

# Broadcast the color [0, 0, 255] to all inactive pixel locations
vis_rgb[inactive_pixels] = [135, 206, 250] 

plt.figure(figsize=(4, 4))
plt.imshow(vis_rgb)
# plt.title("GA Explanation (Blue = Inactive)")
plt.axis('off')
plt.show()




###################################################

def FS(individual, model, target_class):
    global nfe
    # nfe += 1  
    
    img = individual["image"]
    mask = individual["mask"]
    
    # Check if the image has a color channel axis before trying to mean it
    if img.ndim == 3 and img.shape[-1] == 3:
        temp_gray = np.mean(img, axis=2)
    else:
        # If it's already (28, 28) or (28, 28, 1), just ensure it's (28, 28)
        temp_gray = np.squeeze(img)
    
    # Reshape to (1, 28, 28, 1) for the model input
    temp_final = temp_gray[np.newaxis, ..., np.newaxis].astype('float32')
    
    prediction = model(temp_final, training=False).numpy()
    fidelity = prediction[0][target_class]
    
    # Correct sparsity (mask-based)
    sparsity = 1 - np.mean(mask)
    
    return fidelity, sparsity

# Pass that integer into the function
f, s = FS(best_solution, model, target_idx)


print('--- GA Solution ---')
print(f'fidelity: {f:.2f}')
print(f'sparsity: {s:.2f}')


##### fidelity and sparsity for lime

def FS(individual, model, target_class):
    global nfe
    # nfe += 1  

    img = individual["image"]
    mask = individual["mask"]

    # --- FIX: Check dimensions before taking mean ---
    if img.ndim == 3:
        # If it's (28, 28, 3), convert to grayscale (28, 28)
        temp_gray = np.mean(img, axis=2)
    else:
        # It's already (28, 28)
        temp_gray = img

    # Reshape for the model: (1, 28, 28, 1)
    temp_final = temp_gray[np.newaxis, ..., np.newaxis].astype('float32')

    # Prediction
    prediction = model(temp_final, training=False).numpy()
    fidelity = prediction[0][target_class]

    # Sparsity (1 - mean of the binary mask)
    sparsity = 1 - np.mean(mask)

    return fidelity, sparsity
lime_solution = {
    "image": initial_population[9]['image'],
    "mask": initial_population[9]['mask']
}

f_lime, s_lime = FS(lime_solution, model, Y)

print('\n--- LIME Solution ---')
print('fidelity:', np.round(f_lime, 2))
print('sparsity:', np.round(s_lime, 2))
plt.imshow(initial_population[9]['image'])



####################################
####################################
###################  Plot


######################## NFE Comparison #####################
#############################################################



###################################################################




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












###  fitness  for plant




################# Trade  of Digit 5################


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  

# 1. Prepare data for LIME (using your exact variable definitions)
fidelity_lime = [1,1,0.99,0.03,0.09,0.09,0.09,0.09,0.09,0.09]
sparsity_lime = [0.91,0.94,0.93,0.96,0.98,0.98,0.97,0.97,0.97,0.97]

# 2. Prepare data for SAGA
fidelity_saga = 1
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
ax.set_title('Fidelity vs. Sparsity Trade-off: Digit 5', fontsize=32, pad=12)

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
    loc='lower left', 
    fontsize=16, 
    markerscale=0.6,       
    handletextpad=0.8,
    frameon=True
)

plt.tight_layout()
plt.show()


################# Trade  of Digit 8################


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  

# 1. Prepare data for LIME (using your exact variable definitions)
fidelity_lime = [1,0.02,0.04,0.02,0.06,0.0,0.11,0.11,0.11,0.11]
sparsity_lime = [0.8,0.93,0.94,0.94,0.96,0.93,0.97,0.97,0.97,0.97]

# 2. Prepare data for SAGA
fidelity_saga = 1
sparsity_saga = 0.92

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
ax.set_title('Fidelity vs. Sparsity Trade-off: Digit 8', fontsize=32, pad=12)

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
    loc='lower left', 
    fontsize=16, 
    markerscale=0.6,       
    handletextpad=0.8,
    frameon=True
)

plt.tight_layout()
plt.show()

################# Trade  of Digit 4################


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  

# 1. Prepare data for LIME (using your exact variable definitions)
fidelity_lime = [1,0.73,0.59,0.40,0.26,0.29,0.63,0.63,0.63,0.63]
sparsity_lime = [0.89,0.95,0.95,0.97,0.95,0.98,0.98,0.98,0.98,0.98]

# 2. Prepare data for SAGA
fidelity_saga = 0.99
sparsity_saga = 0.95

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
ax.set_title('Fidelity vs. Sparsity Trade-off: Digit 4', fontsize=32, pad=12)

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
    loc='lower left', 
    fontsize=16, 
    markerscale=0.6,       
    handletextpad=0.8,
    frameon=True
)

plt.tight_layout()
plt.show()


################# Trade  of Digit 3################


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  

# 1. Prepare data for LIME (using your exact variable definitions)
fidelity_lime = [1,0.97,0.99,1,0.94,0.94,0.68,0.68,0.68,0.68]
sparsity_lime = [0.86,0.92,0.95,0.92,0.93,0.93,0.97,0.97,0.97,0.97]

# 2. Prepare data for SAGA
fidelity_saga = 0.99
sparsity_saga = 0.95

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
ax.set_title('Fidelity vs. Sparsity Trade-off: Digit 3', fontsize=32, pad=12)

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
    loc='lower left', 
    fontsize=16, 
    markerscale=0.6,       
    handletextpad=0.8,
    frameon=True
)

plt.tight_layout()
plt.show()















import matplotlib.pyplot as plt
import numpy as np

# --- 1. Simulation Data Setup ---
# (Replace these dummy arrays with your actual saved 'best_fitness_history' lists)
# Assuming SAGA + LIME converges faster due to guided initialization
lime_init_history = Digit8LIME
random_init_history = Digit8Random

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

# Keep the view scaled snugly around the length of the longer run
max_iters = max(len(lime_init_history), len(random_init_history))
plt.xlim(-1, max_iters + 1)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="right", fontsize=12, frameon=True, shadow=True)

plt.tight_layout()
plt.show()

