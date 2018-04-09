from BrainDQN_Nature import *

def train(env, brain)
	while True:
	    action = brain.getAction()
	    actionmax = np.argmax(np.array(action))
	    
	    nextObservation,reward,terminal, info = env.step(actionmax)
	    
	    if terminal:
	        nextObservation = env.reset()
	    nextObservation = preprocess(nextObservation)
	    brain.setPerception(nextObservation,action,reward,terminal)

