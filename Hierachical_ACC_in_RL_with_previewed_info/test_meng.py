import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()
eng.Reset(nargout=0)
s = eng.eval('observation')
r = eng.eval('reward')
d = eng.eval('isdone')

s = np.array(s[-1]).astype(np.float32)
r = np.array([[r]]).astype(np.float32)
d = np.array([[d]]).astype(bool)  

print(s, r, d)
print(s.shape, r.shape, d.shape)
eng.eval("set_param(model_name, 'SimulationCommand', 'stop')", nargout=0)
eng.quit()
eng.exit()