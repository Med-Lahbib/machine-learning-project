import torch
from pathlib import Path
p = Path('model/embedding_model.pt')
if not p.exists():
    print('Model file not found:', p)
else:
    state = torch.load(p, map_location='cpu')
    # If it's a checkpoint dict, try to extract state_dict
    if isinstance(state, dict):
        sd = state.get('model_state_dict') or state.get('state_dict') or state
    else:
        sd = None
    if sd is None:
        print('Saved object is not a state_dict; type:', type(state))
    else:
        print('State dict keys and shapes:')
        for k, v in sd.items():
            try:
                print(f"{k}: {tuple(v.shape)}")
            except Exception as e:
                print(f"{k}: (non-tensor) type={type(v)}")
