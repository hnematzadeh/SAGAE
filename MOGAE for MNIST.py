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
####################################################################
####################################################################


from skimage.segmentation import mark_boundaries, slic, quickshift, watershed, felzenszwalb    
from tensorflow.keras.applications import ResNet50, VGG16
import pandas as pd


import copy
# perturbation argument should be array
def perturb_image(img, perturbation, segments,iter):
    global NFE
    active_pixels = np.where(perturbation == 1)[0]
    mask=np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image*mask[:,:,np.newaxis]
    pic=np.expand_dims(perturbed_image, axis=0)
    pred = model.predict(pic)[0][class_to_explain]
    ### fidelity check
    ##for label 1
    # pred = model.predict(pic)[0][class_to_explain][0]
    ### for label 2
    # pred = model.predict(pic)[0][class_to_explain][1]


    max_value=max(model.predict(pic)[0])
    k=model.predict(pic)[0]
    u=k.tolist()
    uu=u.index(max_value)
    Super=perturbation.sum(0)        
    ### fidelity check

    # for label 1 and non-fidelity

    if uu == class_to_explain:
        ac=1
        
    else:
        ac=0
    

    
    
    if Super!=0:
         fit= [(nVar-Super+1)/nVar,pred]
    else:
         fit= [0,0]
    
    

    NFE=NFE+1  
    return pred,perturbed_image,ac,fit 

    

def SinglePointCrossover(x1,x2):
    import random
    import numpy as np
    nVar=len(x1)
    C=random.randint(1,nVar-1)
    y1=(x1[0:C]).tolist() + (x2[C:]).tolist()
    y2=(x2[0:C]).tolist() + (x1[C:]).tolist()
    return y1,y2

def Mutate(x):
    import random
    import numpy as np
    nVar=len(x)
    J=random.randint(0,nVar-1)
    y=copy.deepcopy(x)
    y[J]=1-x[J]
    return y,sum(y)



def Mutate2(x):
    import random
    import numpy as np
    nVar=len(x)
    J=[]
    s=np.trunc(nVar*0.1)
    for i  in range (int(s)):
       J.append(random.randint(0,nVar-1)) 
    y=copy.deepcopy(x)

    for i  in range (int(s)):
        y[J[i]]=1-x[J[i]]
       
    
    return y,sum(y)



def RouletteWheelSelection(P):
    r=random.uniform(0,1)
    c=np.cumsum(P)
    i=np.where(r<np.array(c))[0][0]
    return i
    

def Dominates(x,y):
    
    f1=False
    f2=False
    # flag=all(x >= y for i,j in (x,y))  and  any(x > y for i,j in (x,y))
    c1 = 0 
    c2 = 0
    for i in range(len(x)):
       if(x[i] >= y[i]):
           c1 += 1
       if(x[i] > y[i]):
           c2 += 1

    if(c1 == len(x)):
       f1=True
   
    if(c2 > 0):
       f2=True
   
    flag=f1 and f2   
    return flag

############# Non-dominated sorting function##################
def NonDominatedSorting(initial_pop):
    
    dFrame = pd.DataFrame(index=range(50),columns=range(1))
    nPop=len(initial_pop)
    for i in range(nPop):
        initial_pop[i].DominationSet=[]
        initial_pop[i].DominatedCount=0
        
    F=[]    
    for i in range (nPop):
        for j in range (i+1,nPop):
            p=initial_pop[i]
            q=initial_pop[j]
            if Dominates(p.Fit,q.Fit):
                # p.DominationSet=p.DominationSet.append(j)
                p.DominationSet.append(j)
                q.DominatedCount=q.DominatedCount+1
            if Dominates(q.Fit,p.Fit):
                # q.DominationSet=q.DominationSet.append(i)
                q.DominationSet.append(i)
                p.DominatedCount=p.DominatedCount+1                
            
            initial_pop[i]=p
            initial_pop[j]=q
        
        if initial_pop[i].DominatedCount==0:
          F.append(i)
          initial_pop[i].rank=0
    
    k=0
    dFrame.iloc[k,0]=F

    while(True):
        Q=[]
        for i in dFrame.iloc[k,0]:
            p=initial_pop[i]
            for j in p.DominationSet:
                q=initial_pop[j]
                q.DominatedCount=q.DominatedCount-1
                if q.DominatedCount==0:
                    Q.append(j)
                    q.rank=k+1
                initial_pop[j]=q
        if not Q:
            break;
            
        k=k+1
        dFrame.iloc[k,0]=Q
                    
    return initial_pop,F,dFrame


################  Calculate Distance######################

def CalCrowdingDistance(initial_pop, FrontsList):
    
 for s in range(int(FrontsList.shape[0])):  
   FitVector=[]
   for dd in range (int(len(FrontsList[0][s]))):
      FitVector.append(initial_pop[FrontsList[0][s][dd]].Fit) ###0=i
   nObj=len(FitVector[0])
   n=len(FitVector)
   d=np.zeros((n,nObj))
   for b in range (nObj):
      Fj=[]
      for l in range (n):
          Fj.append(FitVector[l][b])
      yy=np.sort(Fj)
      indexyy=(yy).argsort()[::-1]
      d[indexyy[0],b]=float('inf')
      for o in range (1,n-1):
         d[indexyy[o],b]=abs(yy[o+1]-yy[o-1])/abs(yy[0]-yy[-1])
      d[indexyy[-1],b]=float('inf')
   for i in range (n):
      initial_pop[FrontsList[0][s][i]].CrowdingDistance=np.sum(d[i,:])
         
         
 return initial_pop



################  Sort Population######################

def SortPopulation(initial_pop):
    
    # Sort based on crowding distance
    CD=[]
    RSO=[]
    for i in range (len(initial_pop)):
        CD.append(initial_pop[i].CrowdingDistance)
    # sorted(CD, reverse=True)
    CDSO=np.array(CD).argsort()[::-1]
    import operator
    initial_pop.sort(key=operator.attrgetter('CrowdingDistance'), reverse=True)
    # Sort based on ranks
    for i in range (len(initial_pop)):
        RSO.append(initial_pop[i].rank)
    RSO = np.array(RSO).argsort()
    initial_pop.sort(key=operator.attrgetter('rank'))
      
       
    FitVector = [obj.Fit for obj in initial_pop if obj.rank == 0 ]
    MaxAcc = [obj.pred for obj in initial_pop if obj.rank == 0 ]
    MaximumAcc=max(MaxAcc)
       
    return initial_pop, FitVector,MaximumAcc


##############BinaryTournamentSelection
def   BinaryTournamentSelection(pop):
    I = random.sample(list(range(len(pop))), 2)
    i1=I[0]
    i2=I[1]
    if pop[i1].rank < pop[i2].rank: 
        out=i1
    elif pop[i1].rank > pop[i2].rank:
        out=i2
    elif  pop[i1].CrowdingDistance > pop[i2].CrowdingDistance:
        out=i1
    else:
        out=i2
        
    return out       
    
##############PlotFit Function
def PlotFit(popu):
  fi=popu

  x = [item[0] for item in fi]
  y = [item[1][0] for item in fi]  # Access the first element of the NumPy array

  plt.scatter(x, y, marker='o', color='blue')  # Use scatter plot with circle markers  plt.xlabel('X-Axis Label')  # Replace with your desired label
  plt.ylabel('accuracy')  # Replace with your desired label
  plt.title('number of superpixels')  # Replace with your desired title
  plt.show()


def densifying(chro,prob):
    
    for idx in range (len(chro)):
       
            if chro[idx] == 0:
                if random.random() < prob:
                    chro[idx] = 1
    return chro,sum(chro)

def sparsing(chro,prob):
    
    for idx in range (len(chro)):
       
            if chro[idx] == 1:
                if random.random() < prob:
                    chro[idx] = 0
    return chro,sum(chro)



def deep_copy_struct(initial_struct):
    # Create a new struct with deepcopy for each attribute
    return type(initial_struct)(
        size=copy.deepcopy(initial_struct.size),
        acc=copy.deepcopy(initial_struct.acc),
        chro=copy.deepcopy(initial_struct.chro),
        pred=copy.deepcopy(initial_struct.pred),
        Fit=copy.deepcopy(initial_struct.Fit),
        rank=copy.deepcopy(initial_struct.rank),
        DominationSet=copy.deepcopy(initial_struct.DominationSet),
        DominatedCount=copy.deepcopy(initial_struct.DominatedCount),
        CrowdingDistance=copy.deepcopy(initial_struct.CrowdingDistance),
        im=copy.deepcopy(initial_struct.im)
    )

#############  SYSTEMATIC SEARCH
from ypstruct import struct

WIN=[]
Figure=110
imageLabel=y_test[Figure]  
image=x_test[Figure]
class_to_explain=imageLabel
superpixels= slic(image, n_segments=5,start_label=0)   
nVar=np.unique(superpixels).shape[0]
el=0
TEMP=0
arTEMP=[]
limit=6

heuristic = struct(size=None, acc=None, chro=None, pred=None, Fit=[None,None], rank=None, DominationSet=None, DominatedCount=None, CrowdingDistance=None, im=None)

n = nVar
t=[None]*2**n
lst= [None] * 2**n

for i in range(2**n):
    t[i]=[str(x) for x in bin(i)[2:].zfill(n)]
    t[i]=(np.array(list(t[i]), dtype=int))
    
   
for i in range (2**n):
        active_pixels = np.where(t[i] == 1)[0]    
        mask=np.zeros(superpixels.shape)
        for active in active_pixels:
                mask[superpixels == active] = 1
        perturbed_image = copy.deepcopy(image)
        perturbed_image = perturbed_image*mask[:,:,np.newaxis]
        pic=np.expand_dims(perturbed_image, axis=0)
        pred = model.predict(pic)[0][class_to_explain]
        max_value=max(model.predict(pic)[0])
        k=model.predict(pic)[0]
        u=k.tolist()
        uu=u.index(max_value)      
               
        
        if len(active_pixels) == 0 :
          # fit = (0.7*pred)
          fit = 0
        
        if len(active_pixels) > 0:
          fit=(0.8*pred)+(0.2*((nVar-len(active_pixels)+1)/nVar))
         
          
        if   fit>TEMP and uu == class_to_explain: 
          
                 lst[el]=perturbed_image
                 plt.imshow(perturbed_image)
                 plt.axis("off")
                 plt.show()
                 el=el+1
                 TEMP=fit
                 w=t[i]
                 fit
                 print(t[i],fit)
                 WIN.append(t[i])
                 # print(TEMP)
                 arTEMP.append(TEMP)

############################################
dis=[]
population=[]
paretoSize=[]
best_distance = float('inf') 
distance=best_distance
best_pareto_size = 0
import time
start=time.time()
TEMP=0
imageLabel=y_test[Figure]  
image=x_test[Figure]
class_to_explain=imageLabel
superpixels= slic(image, n_segments=50,start_label=0)   
nVar=np.unique(superpixels).shape[0]
 
### show segmentized image
from skimage.color import gray2rgb
image_rgb = gray2rgb(image)

segmented_image = mark_boundaries(image_rgb, superpixels)
cleaned_image = np.squeeze(segmented_image)
plt.imshow(cleaned_image)
plt.show()

nVar=np.unique(superpixels).shape[0]
precisionP=[]
precisionN=[]
precision=[]
pareto=[]
NFE=len(t)

###### initial parent generation
from ypstruct import struct
FIRST=[None,None]
LAST=[None,None]
nPop=35
pc=0.9;
pm=0.4;
nc=2*round(pc*nPop/2)
nm=round(pm*nPop)
initial_solutions = struct(size=None, acc=None, chro=None, pred=None, Fit=[None,None], rank=None, DominationSet=None, DominatedCount=None, CrowdingDistance=None, im=None)
initial_pop = initial_solutions.repeat(nPop)
best_population = initial_solutions.repeat(nPop)
tagcheck=1
tagflag=1
phi=0.5

import copy

sh=1
for c in range(nPop):
       chromosome= np.random.binomial(1,phi, size=nVar)
       active_pixels = np.where(chromosome == 1)[0]
       print(active_pixels)
       mask=np.zeros(superpixels.shape)
       for active in active_pixels:
               mask[superpixels == active] = 1
       perturbed_image = copy.deepcopy(image)
       perturbed_image = perturbed_image*mask[:,:,np.newaxis]
       pic=np.expand_dims(perturbed_image, axis=0)
       pred = model.predict(pic)[0][class_to_explain]
       print(pred)
       plt.imshow(perturbed_image) 
       [initial_pop[c].im]=[perturbed_image]      
       [initial_pop[c].size]=[len(active_pixels)]
       [initial_pop[c].chro]=[chromosome]
       
       max_value=max(model.predict(pic)[0])
       k=model.predict(pic)[0]
       u=k.tolist()
       uu=u.index(max_value)
       S=len(active_pixels)        
       
       if uu == class_to_explain:
           ac=1
           
       else:
           ac=0
       
       [initial_pop[c].acc] = [ac]
       [initial_pop[c].pred] = [pred]
       if initial_pop[c].size!=0:
            initial_pop[c].Fit= [(nVar-len(active_pixels)+1)/nVar,pred]
       else:
            initial_pop[c].Fit= [0,0]
       
       
       NFE=NFE+1

##########  NonDominatedSorting
[initial_pop, F, FrontsList]=NonDominatedSorting(initial_pop)
FrontsList=FrontsList[~FrontsList.isnull().any(axis=1)]


##########  CrowdingDistance

initial_pop=CalCrowdingDistance(initial_pop, FrontsList)



[initial_pop,ParetoFront,MaximumAcc]=SortPopulation(initial_pop)
MaximumAccuracy=MaximumAcc

preds = model.predict(x_test)
distance=preds[Figure, class_to_explain]/ MaximumAcc


# distance=model.predict(x_test)[Figure][class_to_explain][0]/ MaximumAcc
dis.append(distance)
best_population=initial_pop
paretoSize.append(len(ParetoFront))
print("distance", distance, 'tagcheck', tagcheck)
print ("Iteration",  "  ", 0,"      ","Number of F0 members", "  ",   len(ParetoFront)  )
pareto.append(len(ParetoFront))
precision.append(MaximumAccuracy)


import random
############### NSGA-II main loop
MaxIt=100
probability=0.5
alpha=0.2
for es in range (MaxIt):
   
   popc1=initial_solutions.repeat(int(nc/2))
   popc2=initial_solutions.repeat(int(nc/2)) 
   Xover=list(zip(popc1,popc2))
   for k in range (int(nc/2)):
       
       
       # Select First Parent
       # i1=RouletteWheelSelection(P)
       i1=BinaryTournamentSelection(initial_pop)
       # i1=random.randint(0,nPop-1)
       p1=initial_pop[i1].chro
       # Select Second Parent
       # i2=RouletteWheelSelection(P)
       i2=BinaryTournamentSelection(initial_pop)
       # i2=random.randint(0,nPop-1)
       p2=initial_pop[i2].chro
       #Apply Crossover
       Xover[k][0].chro,Xover[k][1].chro=np.array(SinglePointCrossover(p1,p2))
       #Evaluate Offspring
       Xover[k][0].pred,Xover[k][0].im,Xover[k][0].acc,Xover[k][0].Fit=perturb_image(image,Xover[k][0].chro,superpixels,es)
       Xover[k][0].size=Xover[k][0].chro.sum(0)
       
       Xover[k][1].pred,Xover[k][1].im,Xover[k][1].acc, Xover[k][1].Fit=perturb_image(image,Xover[k][1].chro,superpixels,es)
       Xover[k][1].size=Xover[k][1].chro.sum(0)
       popc=initial_solutions.repeat(nc)
   i=0
   for s in range (len(Xover)):
       for j in range(2):
            popc[i]=Xover[s][j]
            i=i+1
    
###### mutation


   popm=initial_solutions.repeat(nm)
   

###########Adaptive Bit Flip mutation     
   
   if distance>3 and tagcheck==10:
               
                   probability= probability + alpha *(1- probability) 
                   print("distance", distance, 'tagcheck', tagcheck)
                   print(probability)
                   print('Densifying')
                   for k in range(nm):
   
                        i=random.randint(0,nPop-1)
                        p=initial_pop[i].chro
                        popm[k].chro,popm[k].size = densifying(p, probability)
                        popm[k].pred,popm[k].im,popm[k].acc,popm[k].Fit=perturb_image(image,popm[k].chro,superpixels,es)
                
                   tagcheck=0
               
   elif distance>3 and tagcheck<10: 
                print("distance", distance, 'tagcheck', tagcheck)

                # tagcheck=tagcheck+1 
                for k in range(nm):
   
                    i=random.randint(0,nPop-1)
                    p=initial_pop[i].chro
                    popm[k].chro,popm[k].size=Mutate2(p)
                    popm[k].pred,popm[k].im,popm[k].acc,popm[k].Fit=perturb_image(image,popm[k].chro,superpixels,es)
   elif distance<=3  and tagflag==10:
        print("distance", distance, 'tagflag', tagflag)

        probability = probability - alpha * probability
        # probability = 0.7
        print(probability)
        print('Sparsing')

        for k in range(nm):

            i=random.randint(0,nPop-1)
            p=initial_pop[i].chro
            popm[k].chro,popm[k].size = sparsing(p, probability)
            popm[k].pred,popm[k].im,popm[k].acc,popm[k].Fit=perturb_image(image,popm[k].chro,superpixels,es)
        
        tagflag=0
      
       
   elif distance<=3 and tagflag<10:
         print("distance", distance, 'tagflag', tagflag)
      
      
         for k in range(nm):

              i=random.randint(0,nPop-1)
              p=initial_pop[i].chro

              popm[k].chro,popm[k].size=Mutate2(p)

              popm[k].pred,popm[k].im,popm[k].acc,popm[k].Fit=perturb_image(image,popm[k].chro,superpixels,es)
       
       
#######  merge population        
   initial_pop= initial_pop+popc+popm    
######  SORT

####### FrontsList contain the fronts' members including paretofront and the rest
   [initial_pop, F, FrontsList]=NonDominatedSorting(initial_pop)
   FrontsList=FrontsList[~FrontsList.isnull().any(axis=1)]
########Crowding Distance Calculation#################  
   initial_pop=CalCrowdingDistance(initial_pop, FrontsList)
#########Sort population based on rank and crowding distance#################     
   [initial_pop, ParetoFront, MaximumAcc]=SortPopulation(initial_pop)
####Truncate
   initial_pop=initial_pop[0:nPop]
   # print("len initial_pop", " ", len(initial_pop))

   FF=[]
   M=[]
   
   for dd in range (len(initial_pop)):
         if (initial_pop[dd].rank==0) :
            FF.append(initial_pop[dd].Fit)
            M.append(initial_pop[dd].pred)
   if len(M)!=0:
        MaximumAcc=max(M)
   # print("Len pareto front", " ", len(FF))
    
   
    
   preds = model.predict(x_test)
   distance=preds[Figure, class_to_explain]/ MaximumAcc
   # distance=model.predict(x_test)[Figure][class_to_explain][0]/MaximumAcc
   paretoSize.append(len(FF))

   
   if  distance> 3 and abs(distance-dis[-1])==0:
       tagcheck=tagcheck+1
       # tagflag=1
       
   if distance> 3 and abs(distance-dis[-1])!=0:
       tagcheck=1
       # tagflag=1
   if  distance <= 3 and abs(distance-dis[-1])==0:
      tagflag=tagflag+1
       # tagcheck=1
   if  distance <= 3 and abs(distance-dis[-1])!=0:
       tagflag=1
       # tagcheck=1
   dis.append(distance)
   if (distance < best_distance) or (distance == best_distance and len(FF) > best_pareto_size):
        best_distance = distance
        best_population = [deep_copy_struct(sol) for sol in initial_pop]

        best_pareto_size = (len(FF))
   pareto.append(len(FF))
   

   print ("Iteration",  "  ", es+1,"      ","Number of F0 members", "  ",  len(FF)   )
end=time.time()
print("Time = ", end-start,  "NFE = ", NFE)







##### remove those solutions in pareto front with of initial_pop that the  acc=0
new_initial_pop = [solution for solution in best_population if solution.rank == 0 and solution.acc == 1]
# new_initial_pop = [solution for solution in best_population if solution.rank == 0  ]

# new_initial_pop=initial_pop
######  remove identical solutions in ParetoFront
ip = struct(size=None, acc=None, pred=None, Fit=[None,None], rank=None, DominationSet=None, DominatedCount=None, CrowdingDistance=None, im=None)
unique_solutions = ip.repeat(len(new_initial_pop))
templist=[]

# Iterate through 'initial_pop' and filter duplicates based on fitness vectors
for c in range (len(new_initial_pop)):
    fitness_vector = new_initial_pop[c].Fit  # Convert the fitness vector to a tuple
    
    # Check if the fitness vector is unique
    if fitness_vector not in templist:
        print(c)
        unique_solutions[c] = new_initial_pop[c]
        templist.append(fitness_vector)
t1=[]
cc=0
for dd in range (len(unique_solutions)):
    if (unique_solutions[dd].Fit != [None,None]):
        plt.imshow(unique_solutions[dd].im)
        plt.title(dd+1)
        plt.axis('off')
        plt.show()
        t1.append(unique_solutions[dd].im)
        cc=cc+1
        print(unique_solutions[dd].Fit)

######## In case you want to see the optimal images individually
# plt.imshow(t1[0])
# plt.axis("off")







#### VOTING
mask = np.any(t1[0] != [0, 0, 0], axis=-1)
pixel_counts = np.zeros_like(mask, dtype=np.uint16)
for img in t1:
    pixel_mask = np.any(img != [0, 0, 0], axis=-1)
    pixel_counts += pixel_mask
final_mask = pixel_counts >= (len(t1)//2+1)
# final_mask = pixel_counts == (len(t1))
# 

final_mask_3d = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.float32)
final_mask_3d[final_mask] = 1
plt.imshow(image*final_mask_3d)
plt.axis('off')
#########   In case you want to try the UNION of optimal images instead of majority voting

union_mask = np.zeros(t1[0].shape[:2], dtype=bool)
# Combine masks by taking the union
for img in t1:
    pixel_mask = np.any(img != [0, 0, 0], axis=-1)
    union_mask = np.logical_or(union_mask, pixel_mask)

# Convert the union mask to a 3D representation for visualization
union_mask_3d = np.zeros((*union_mask.shape, 3), dtype=np.float32)
union_mask_3d[union_mask] = 1

# Assuming 'image' is the original image you want to overlay the mask on
plt.imshow(image * union_mask_3d)
plt.axis('off')
plt.show()
final_mask_3d=union_mask_3d




#####CREATING BASE image
# BASE= np.all(WIN, axis=0).astype(int)
BASE=w
active_pixels = np.where(BASE == 1)[0]
print(active_pixels)
superpixels= slic(image, n_segments=5, start_label=0)   
mask=np.zeros(superpixels.shape)
for active in active_pixels:
        mask[superpixels == active] = 1
perturbed_image = copy.deepcopy(image)
perturbed_image = perturbed_image*mask[:,:,np.newaxis]
plt.imshow(perturbed_image, cmap = 'gray')
plt.axis("off")

######### Generating final  image with BLACK background
black_back=np.where(perturbed_image != 0, image*final_mask_3d, 0)
plt.imshow(black_back)
plt.axis('off')


###########   Fidelity of the black image


import numpy as np

# 1. Start with your black_back (the grayscale version)
# We need to make sure it is (28, 28, 1)
if black_back.ndim == 2:
    # If it's just (28, 28), add the channel
    prediction_input = black_back[:, :, np.newaxis]
elif black_back.ndim == 3 and black_back.shape[-1] == 3:
    # IF your black_back accidentally became RGB, convert it back to grayscale
    prediction_input = np.mean(black_back, axis=-1, keepdims=True)
else:
    prediction_input = black_back

# 2. Add the Batch dimension to make it (1, 28, 28, 1)
prediction_input = np.expand_dims(prediction_input, axis=0)

# 3. Double check the shape! 
# It MUST be (1, 28, 28, 1) for a standard MNIST model.
print(f"Feeding shape to model: {prediction_input.shape}") 

# 4. Predict
prediction = model.predict(prediction_input)
score = np.round(prediction[0][class_to_explain],2)

print(f"Success! Score for class {class_to_explain}: {score}")

################################################################
######################## sparsity #############################
def FS(individual, model, target_class, mask):
    global nfe
    # nfe += 1  

    

    temp_gray = np.mean(individual, axis=2)
    temp_final = temp_gray[np.newaxis, ..., np.newaxis].astype('float32')

    prediction = model(temp_final, training=False).numpy()
    fidelity = prediction[0][target_class]

    # ✅ Correct sparsity (mask-based)
    sparsity = 1 - np.mean(mask)

    return fidelity, sparsity


binary_mask = np.any(black_back != 0, axis=-1).astype(int)
f, s = FS(black_back, model, 8, binary_mask)
# f, s = FS(refined_solution, model, Y)

# fitness = 0.6 * f + 0.4 * s

# print('--- GA Solution ---')
print('fidelity:', np.round(f, 2))
print('sparsity:', np.round(s, 2))
# print('fitness :', np.round(fitness, 2))



 
###########################################################
####################### lavender background ###############

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
import copy

# --- Your existing logic to get perturbed_image ---
BASE = w
active_pixels = np.where(BASE == 1)[0]
superpixels= slic(image, n_segments=5, start_label=0)   
mask = np.zeros(superpixels.shape)
for active in active_pixels:
    mask[superpixels == active] = 1

perturbed_image = copy.deepcopy(image)
# Ensuring mask has the correct 3D shape for multiplication
perturbed_image = perturbed_image * mask[:, :, np.newaxis]
# plt.imshow(perturbed_image, cmap = 'gray')
# plt.axis("off")
# --- NEW: Creating the Green Background Intersection ---

# 1. Helper to force arrays into (H, W, 3) shape
def force_rgb(img):
    img = np.squeeze(img) # Remove singleton dimensions like (28, 28, 1, 1)
    if img.ndim == 2:
        return gray2rgb(img)
    elif img.ndim == 3 and img.shape[-1] == 1:
        return gray2rgb(img.squeeze())
    return img

# 2. Convert perturbed_image and your final_mask to RGB
perturbed_rgb = force_rgb(perturbed_image)
# If final_mask_3d is what you want to intersect with perturbed_image:
mask_to_intersect = force_rgb(binary_mask)

# 3. Create the intersection (Foreground)
# This keeps only pixels present in BOTH the perturbed image and the mask
intersection_foreground = perturbed_rgb * (mask_to_intersect > 0)

# Check if image is float (0-1.0) or int (0-255)
is_float = perturbed_rgb.max() <= 1.0

# Define Light Purple Color: [Red, Green, Blue]
# A soft lavender mix:
light_purple = [0.9, 0.8, 1.0] if is_float else [230, 204, 255]
purple_bg = np.full_like(perturbed_rgb, light_purple)

# 5. Combine: If pixel is in intersection, show it. Otherwise, show light purple.
condition = np.max(intersection_foreground, axis=-1, keepdims=True) > 0
light_purple_final = np.where(condition, intersection_foreground, purple_bg)

# 6. Plot
plt.imshow(light_purple_final)
# plt.title("MOGAE (Digit 3)")
plt.axis('off')
plt.show()

#################################################################



explanations= []
explanations.append(S1)
explanations.append(S2)
explanations.append(S3)

x2 = x_test[110]
y2 = y_test[110]


####################  Consistency #########################
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
            print(f"Jaccard(E_{i+1}, E_{j+1}) = {score:.4f}")

    # Final Stability Score
    stability = np.mean(jaccard_scores)
    print(f"\nFinal Stability Score: {stability:.4f}")
    
    return stability

# Example Usage:
stability_score = evaluate_stability(x2, model, y2, n_runs=3)
np.round(stability_score, 2)




















