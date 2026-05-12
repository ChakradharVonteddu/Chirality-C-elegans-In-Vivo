from ..models import ModelAB, ModelES, ModelABp2, ModelESp2, ModelESFriction

#            ABal, ABar, ABpr, ABpl
#INIT = (0.7, 0.5, 0, 0.7, -0.5, 0, -0.3, -0.5, 0, -0.3, 0.5, 0)
INIT = (0.65, 0.45, 0.25, 0.65, -0.45, 0.25, -0.25, -0.45, 0.25, -0.25, 0.45, 0.25)

#====================== 2 parameter models ======================
# MODEL AB
# GET_VELOCITY = ModelAB.get_velocity

# MODEL EXTENDING SPRING
# GET_VELOCITY = ModelES.get_velocity

# MODEL ABP2
# GET_VELOCITY = ModelABp2.get_velocity

# MODEL EXTENDING SPRING P2
GET_VELOCITY = ModelESp2.get_velocity

# MODEL EXTENDING SPRING, Cell Wall Velocity
# GET_VELOCITY = ModelESFriction.get_velocity