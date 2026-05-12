# Cell-Chirality-Model

Python Implementation of Cell Chirality Spring Model in early embryonic development of C. Elegans.

Notes: https://www.overleaf.com/read/xqqjsrdqwkcj these set of notes contains the equation construction of the basic constant spring model, which is used to derive all further models.

## Current Models

The following are the models that are fitted. Any model with p2 considers the p2 cell, and the dash after denotes where the p2 cell is fixed (0 degrees, 30 degrees, 45 degrees...). The Constant Spring models have springs with constant rest lengths, which the Extending Spring models have variable spring lengths that scale to 1.5 of the original rest length by the end of the experiment. The Extending Spring models also consider an ellpsoid eggshell that pushes back when the cells get too close. The all prefix distinguishes between models where all springs expand verses models where only the springs between the cells undergoing mitosis expands.

CS (constant spring) Models:

1. AB
2. ABp2-0, ABp2-30, ABp2-45

ES (extending spring) Models:

1. ES, allES
2. ESp2-0, ESp2-30, ESp2-45
3. allESp2-0, allESp2-30, allESp2-45

So far, ESp2-45(A=0.1355238,B=0.02709549) and ABp2-45(A=0.1355238,B=0.02709549) have been the most promising candidates. Their fit are included in the "Model Fit" section below. Also see visualizations and plots in "figures". 

## Further Models

Further model candidates include:

- on / off friction: a variable of current models where the friction is only activated when the cells are in contact with an object of interest
- p2 wall: p2 cell is currently modelled as a cell fixed in space that is not spinning. Another possible set of models model the p2 as a wall
- fit pushback: currently, the force with which the egg shell of Extended Spring models push back is modelled as a linear spring force that shares a spring constant with the intracell springs. A possible path to explore would be to let this linear force be a 3rd parameter to fit.

## Important Design Decisions

This section goes over important notes about the code base and design decisions that should be revisited and perhaps reconsidered.

- models use RK45 to resolve ODE, and the default fitting alg of scipy.curve_fit for fitting
- all models have diagonal springs that activate conditionally
- the ABp cells have a spring with the p2 cell. For ES models, even though the springs expand, this spring with the p2 cell is fixed.
- all ES models retain the spring between cells and the "EMS" (z=-0.5 plane). This makes the model a more complex but improves the dorsal fit
- egg shell is tight (e0=3/2 or 2, e1 = 1)
- the residuals of extended spring models still tries to keep the distance between all adjancent cells as 1
- the models themselves have the directions worked out and only requires that the parameters be the suitable magnitudes. Thus, the fit for parameters is bounded to be >= 0
- the initial parameter guess is a vector of all 0s
- the data is actually not great

## Model Fit
### ABp2
![CS ABp2 Fit](https://github.com/YouTelllMe/Cell-Chirality-Model/blob/main/figures/CSp2/fit.png)
### ESp2
![ES ESp2 Fit](https://github.com/YouTelllMe/Cell-Chirality-Model/blob/main/figures/ESp2/fit.png)

