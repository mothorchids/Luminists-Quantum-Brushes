import numpy as np
import importlib.util
import pennylane as qml

spec = importlib.util.spec_from_file_location("utils", "effect/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

spec_steer = importlib.util.spec_from_file_location("steer", "effect/steerable/helper.py")
steer = importlib.util.module_from_spec(spec_steer)
spec_steer.loader.exec_module(steer)

"""
Utility functions for colors
"""
def selection_to_state(image, region, nb_controls):
    pixels = image[region[:, 0], region[:, 1]] # RGBA 
    print(f"initial pixels {pixels}")
    pixels = pixels.astype(np.float32) / 255.0

    U, S, Vt = np.linalg.svd(pixels, full_matrices=False)
    S_safe = np.clip(S, 1e-30, None)  # Avoid log(0) or negative
    log_s = np.log(S_safe)
    if nb_controls == 2:
        return U, S, Vt, log_s / np.linalg.norm(log_s)

    # state = Vt.flatten() # 16 entries
    if nb_controls == 3:
        log_s2 =np.concatenate([log_s, Vt @ log_s])
        
        return U, S, Vt, log_s2/np.linalg.norm(log_s2)
    elif nb_controls == 4:
        # return U, S, Vt, state / np.linalg.norm(state)
        ### First method
        # log_s4 = (np.kron(log_s.reshape(-1, 1), log_s.reshape(-1,1).T)).flatten()
        ### Second method
        log_s4 = np.concatenate([log_s, Vt @ log_s, Vt @ Vt @ log_s, Vt @ Vt @ Vt @ log_s])
        return U, S, Vt, log_s4/np.linalg.norm(log_s4)

    else :
        raise ValueError(f"Unsupported number of controls: {nb_controls}")

def state_to_pixels(U, S, Vt, state):
    """
    template : selection of pixels from an image
    state : output state from circuit
    """
    state = np.array(state)
    nb = len(state)
    S_new = np.copy(np.diag(S))
    Vt_new = np.copy(Vt)
    S_safe = np.clip(S, 1e-30, None)  # Avoid log(0) or negative
    log_s = np.log(S_safe)
    norm_log_s = np.linalg.norm(log_s)
    if nb==4:
        exponent = np.clip(norm_log_s * state, -700, 700) # to avoid overflow
        S_new = np.diag(np.exp(exponent))
    elif nb==8 :
        op = np.eye(8)
        op[4:, 4:] = Vt_new
        state_new = (np.linalg.inv(op) @ state)[:4]
        exponent = np.clip(norm_log_s * state_new/np.linalg.norm(state_new), -700, 700) # to avoid overflow
        S_new = np.diag(np.exp(exponent))
    elif nb==16:
        ### First method
        # def best_self_outer_complex(M):
        #     H = 0.5 * (M + M.conj().T)   # Hermitian part
        #     lam, U = np.linalg.eigh(H)   # real eigenvalues
        #     lambda1 = lam[-1]
        #     u1 = U[:, -1]
        #     alpha = np.sqrt(max(lambda1, 0.0))
        #     a = alpha * u1
        #     A = np.outer(a, a.conj())    # a a^H
        #     res_norm = np.linalg.norm(M - A, ord='fro')
        #     return a, A, res_norm 
        # state_new, _, _ = best_self_outer_complex(state.reshape(4, 4))
        ### Second method
        def block_diag_np(*mats):
            # Determine total size
            sizes = [m.shape[0] for m in mats]
            total = sum(sizes)

            # Allocate zero matrix
            out = np.zeros((total, total), dtype=mats[0].dtype)

            # Fill blocks
            offset = 0
            for m in mats:
                n = m.shape[0]
                out[offset:offset+n, offset:offset+n] = m
                offset += n

            return out
        op = block_diag_np(np.eye(4), Vt_new, Vt_new @ Vt_new, Vt_new @ Vt_new @ Vt_new)
        state_new = (np.linalg.inv(op) @ state)[:4]

        exponent = np.clip(norm_log_s * state_new/np.linalg.norm(state_new), -700, 700) # to avoid overflow
        S_new = np.diag(np.exp(exponent))
    else :
        raise ValueError(f"Unsupported number of param in state : {nb}")
    print(f"========== Output ==============\n U=\n{U},\n S=\n{S_new},\n Vt=\n{Vt_new}")
    return U @ S_new @ Vt_new
    
"""
Measurement
"""
def create_circuit_and_measure(params, source, target, initial, n_qubits):
    t = params["user_input"]["t"]
    n_steps = params["user_input"]["timesteps"]
    dev = qml.device("default.qubit", wires=n_qubits)
    circuit = steer.build_circuit(dev, params["user_input"], source, target, n_qubits)
    output = circuit(initial_state=initial, n_qubits=n_qubits, t=t, n_steps=n_steps, n=1)
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
    print(f"There are {len(clicks)} clicks.")
    print(clicks)
    assert len(clicks) <= 3, "The number of clicks must be 3, i.e. source, target, paste"
    
    
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
    U_s, S_s, Vt_s, state_s = selection_to_state(image, region_s, nb_controls)
    print("=== Computing angles from target ===")
    _, _, _, state_t = selection_to_state(image, region_t, nb_controls)

    # Comput brush effects
    output_measures = create_circuit_and_measure(params, state_s, state_t, state_s, nb_controls).real
    print(f"input state: {state_s}\n output state: {output_measures}")

    # Apply effects
    region_output = region_s
    U_o, S_o, Vt_o = U_s, S_s, Vt_s
    region_paste = region_s
    if not params["user_input"].get("Souce=Paste", True):
        assert len(clicks)==3, "At least 3 clicks are required for paste different from source"
        if len(paths[2])>10:
            region_output = utils.points_within_lasso(paths[2], border = (height, width))
            region_paste = region_output
            pixels = image[region_output[:, 0], region_output[:, 1],:]
            pixels = pixels.astype(np.float32) / 255.0
            U_o, S_o, Vt_o = np.linalg.svd(pixels, full_matrices=False)
        else :
            barycenter = np.rint(paths[0].mean(axis=0)).astype(int)
            offset = clicks[2]-barycenter
            print(offset)
            region_paste = utils.points_within_lasso(paths[0]+offset, border = (height, width))
            print(region_paste)
        
    new_pixels = state_to_pixels(U_o, S_o, Vt_o, output_measures)
    # Clip RGB channels (first 3 columns) to 0–255
    new_pixels[:, :3] = np.clip(new_pixels[:, :3], 0, 255)
    # Clip alpha channel (fourth column) to 0–50
    new_pixels[:, 3] = np.clip(new_pixels[:, 3], 50, 255)
    print(f"new pixels: {new_pixels}")
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
