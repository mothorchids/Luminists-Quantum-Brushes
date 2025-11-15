import numpy as np
import importlib.util
import jax
import jax.numpy as jnp
import pennylane as qml

spec = importlib.util.spec_from_file_location("utils", "effect/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

spec_steer = importlib.util.spec_from_file_location("steer", "effect/steerable/steer_training.py")
steer = importlib.util.module_from_spec(spec_steer)
spec_steer.loader.exec_module(steer)

"""
Utility functions for colors
"""
def selection_to_state(image, region, nb_controls):
    pixels = image[region[:, 0], region[:, 1]] # RGBA 
    print(f"initial pixels {pixels}")
    pixels = pixels.astype(np.float32) / 255.0

    _, S, Vt = np.linalg.svd(pixels, full_matrices=False)
    log_s = np.log(S)
    if nb_controls == 2:
        return log_s / np.linalg.norm(log_s)

    state = Vt.flatten() # 16 entries
    if nb_controls == 3:
        log_s_normalized = log_s / np.linalg.norm(log_s)
        state_normalized = state[:4] / np.linalg.norm(state[:4])
        return jnp.concatenate([0.5*log_s_normalized , 0.5*state_normalized])
    elif nb_controls == 4:
        return state / np.linalg.norm(state)
    else :
        raise ValueError(f"Unsupported number of controls: {nb_controls}")

def state_to_pixels(template, state):
    """
    template : selection of pixels from an image
    state : output state from circuit
    """
    nb = len(state)
    state = np.abs(state) # complexe->real (take magnitude)
    U, S, Vt = np.linalg.svd(template, full_matrices=False)
    S_new = np.copy(np.diag(S))
    Vt_new = np.copy(Vt)
    log_s = np.log(S)
    norm_log_s = np.linalg.norm(log_s)
    if nb==4:
        S_new = np.diag(np.exp(norm_log_s*state))
    elif nb==8 :
        S_new = np.diag(np.exp(norm_log_s*state[:4]))
        vt = Vt.flatten() 
        vt_norm = np.linalg.norm(vt[:4])
        vt_modified =  vt_norm * state[4:]
        Vt_new = np.concatenate([vt_modified, vt[4:]]).reshape(Vt.shape)
        # Vt_new = (vt_modified+vt[4:]).reshape(Vt.shape)
    elif nb==16:
        vt_norm = np.linalg.norm(Vt)
        Vt_new = (vt_norm * state).reshape(Vt.shape)
    else :
        raise ValueError(f"Unsupported number of param in state : {nb}")
    print(f"========== Output ==============\n U={U}, S={S_new}, Vt={Vt_new}")

    return U @ S_new @ Vt_new
    
"""
Measurement
"""
def create_circuit_and_measure(params, source, target, initial, n_qubits):
    t = params["user_input"]["t"]
    n_steps = params["user_input"]["timesteps"]
    dev = qml.device("default.qubit", wires=n_qubits)
    circuit = steer.build_circuit(dev, params["user_input"], source, target, n_qubits)
    output = circuit(initial, n_qubits, t, n_steps=n_steps, n=1)
    return output

"""
Brush execution 
"""
def run(params):
    """
    Executes the effect pipeline based on the provided parameters.

    Args:
        parameters (dict): A dictionary containing all the relevant data.

    Returns:
        Image: the new numpy array of RGBA values or None if the effect failed
    """
    
    # Extract image to work from
    image = params["stroke_input"]["image_rgba"]
    # It's a good practice to check any of the request variables
    assert image.shape[-1] == 4, "Image must be RGBA format"

    height = image.shape[0]
    width = image.shape[1]

    # Extract the copy and past points
    clicks = params["stroke_input"]["clicks"]
    print(clicks)
    assert len(clicks) <= 3, "The number of clicks must 3, i.e. source, target, paste"
    print(f"There are {len(clicks)} clicks.")
    
    # Extract the lasso path
    path = params["stroke_input"]["path"]

    start_coordinates = [
        np.where((path == click).all(axis=1))[0][0]
        for click in clicks
    ]

    paths = []
    for i in range(len(clicks)-1):
        paths.append(path[start_coordinates[i]:start_coordinates[i+1]])
        print(f"path_{i} starts from {start_coordinates[i]} and has length {len(paths[i])}")
    paths.append(path[start_coordinates[len(clicks)-1]:])
    print(f"path_{len(clicks)-1} starts from {start_coordinates[len(clicks)-1]} and has length {len(paths[len(clicks)-1])}")

    nb_controls = params["user_input"]["Controls"]

    # Create the regions (source and target)
    region_s = utils.points_within_lasso(paths[0], border = (height, width))
    print("region_s============",len(region_s))
    region_t = utils.points_within_lasso(paths[1], border = (height, width))

    # Encode colors to probability states
    print("=== Computing angles from source ===")
    state_s = selection_to_state(image, region_s, nb_controls)
    print(state_s)
    print("=== Computing angles from target ===")
    state_t = selection_to_state(image, region_t, nb_controls)

    # Comput brush effects
    output_measures = create_circuit_and_measure(params, state_s, state_t, state_s, nb_controls)

    # Apply effects
    region_paste = region_s
    if not params["user_input"].get("Souce=Paste", True):
        assert len(clicks)==3, "At least 3 clicks are required for paste different from source"
        if len(paths[2])>10:
            region_paste = utils.points_within_lasso(paths[2], border = (height, width))
        else :
            barycenter = np.rint(paths[0].mean(axis=0)).astype(int)
            offset = clicks[2]-barycenter
            print(offset)
            region_paste = utils.points_within_lasso(paths[0]+offset, border = (height, width))
            print(region_paste)
        
    pixels = image[region_paste[:, 0], region_paste[:, 1],:]
    pixels = pixels.astype(np.float32) / 255.0
    new_pixels = state_to_pixels(pixels, output_measures)
    image[region_paste[:, 0], region_paste[:, 1],:] = (new_pixels * 255).astype(np.uint8)

    # Optional: show source and target regions
    if params["user_input"].get("show source & target", False):
        # highlight source
        outline = utils.points_within_radius(paths[0], radius=30, border = None, return_distance = False)
        outline_color = image[outline[:, 0], outline[:, 1]].astype(np.float32)/255
        outline_color[...,:3] = params["user_input"]["show color"] / 255 # Set RGB channels
        outline_color[...,3] = 1 # alpha in RGBA 
        image[outline[:, 0], outline[:, 1]] = utils.apply_patch_to_image(image[outline[:, 0], outline[:, 1]], outline_color)
        # highlight target
        outline = utils.points_within_radius(paths[1], radius=30, border = None, return_distance = False)
        outline_color = image[outline[:, 0], outline[:, 1]].astype(np.float32)/255
        outline_color[...,:3] = params["user_input"]["show color"] / 255 # Set RGB channels
        outline_color[...,3] = 1 # alpha in RGBA 
        image[outline[:, 0], outline[:, 1]] = utils.apply_patch_to_image(image[outline[:, 0], outline[:, 1]], outline_color)
 
    return image
