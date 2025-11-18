#!/usr/bin/env python3
import os
import argparse
import json

import numpy as np
import pennylane as qml

import helper as steer
import utils

from PIL import Image

def define_region(img, coord1=None, coord2=None):
    """
    Define a rectangular region in an image given two coordinates.

    Parameters:
        img (np.array): Input image (2D or 3D array).
        coord1 (tuple or None): (row, col) of the first corner.
        coord2 (tuple or None): (row, col) of the opposite corner.

    Returns:
        np.array: Array of coordinates within the defined region.
                  Each row is (row, col). If coordinates are None, uses the full image.
    """
    # If coordinates are not provided, use the full image
    if coord1 is None or coord2 is None:
        r1, c1 = 0, 0
        r2, c2 = img.shape[0]-1, img.shape[1]-1
    else:
        r1, c1 = coord1
        r2, c2 = coord2
    
    # Ensure coordinates are within image bounds
    r1, r2 = max(0, min(r1, img.shape[0]-1)), max(0, min(r2, img.shape[0]-1))
    c1, c2 = max(0, min(c1, img.shape[1]-1)), max(0, min(c2, img.shape[1]-1))
    
    # Get min/max to handle any order of coordinates
    r_start, r_end = min(r1, r2), max(r1, r2)
    c_start, c_end = min(c1, c2), max(c1, c2)
    
    # Create coordinates
    rows, cols = np.meshgrid(np.arange(r_start, r_end+1), np.arange(c_start, c_end+1), indexing='ij')
    coords = np.stack([rows.ravel(), cols.ravel()], axis=-1)
    
    return coords

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

def create_circuit_and_measure_multiple(params, source, target, initial, n_qubits, ts):
    n_steps = params["user_input"]["timesteps"]
    dev = qml.device("default.qubit", wires=n_qubits)
    circuit = steer.build_circuit(dev, params["user_input"], source, target, n_qubits)
    output = np.empty(16, dtype=object)
    for k in range(len(ts)):
        output[k] = circuit(initial_state=initial, n_qubits=n_qubits, t=ts[k], n_steps=n_steps, n=1)
    return output

def run(params,ts=None):

    image_s = params["stroke_input"]["image_s_rgba"]
    assert image_s.shape[-1] == 4
    image_t = params["stroke_input"]["image_t_rgba"]
    assert image_t.shape[-1] == 4
    nb_controls = params["user_input"]["Controls"]
    region_s = define_region(image_s)
    region_t = define_region(image_t)

    # Encode colors to probability states
    print("=== Computing angles from source ===")
    U_s, S_s, Vt_s, state_s = selection_to_state(image_s, region_s, nb_controls)
    print(f"state_s={state_s}")
    print("=== Computing angles from target ===")
    _, _, _, state_t = selection_to_state(image_t, region_t, nb_controls)
    print(f"state_t={state_t}")

    output_measures = create_circuit_and_measure_multiple(params, state_s, state_t, state_s, nb_controls,ts).real
    
    region_output = region_s

    pixels = image_s[region_output[:, 0], region_output[:, 1], :]
    pixels = pixels.astype(np.float32) / 255.0

    new_images = np.empty(16, dtype=object)
    for k in range(len(ts)):
        print(f"output state: {output_measures[k].real}")
        new_pixels = state_to_pixels(U_s, S_s, Vt_s, output_measures[k].real)
        new_images[k] = (new_pixels * 255).astype(np.uint8)

    # return new_image
    return new_images

def main():
    parser = argparse.ArgumentParser(description="Run steerable quantum image effect.")
    parser.add_argument("-i1", "--image1", help="First input image (source)")
    parser.add_argument("-i2", "--image2", help="Second input image (target)")
    parser.add_argument("-p", "--params", default="config.json", help="JSON params file")
    parser.add_argument("-C", "--Controls", type=int, default=2, help="Number of controls (2, 3, or 4)")
    parser.add_argument("-t", type=float, default=0.0, help="Time parameter for the circuit")
    parser.add_argument("-o", "--output", default="output.png", help="Output image path")
    
    args = parser.parse_args()

    # Load params JSON
    with open(args.params, "r") as f:
        params = json.load(f)

    # Load two input images
    img1 = np.array(Image.open(args.image1).convert("RGBA"))
    img2 = np.array(Image.open(args.image2).convert("RGBA"))

    # Change params if  precised by user
    params["stroke_input"]["image_s_rgba"] = img1
    params["stroke_input"]["image_t_rgba"] = img2   # optional if needed by user later
    params["user_input"]["Controls"] = args.Controls
    
    # Define time steps
    t_start = params["user_input"].get("t_start", 0.0)
    t_end = params["user_input"].get("t_end", 1.0)
    t_number_of_images = params["user_input"].get("t_number-of-images", 8)
    ts = list(np.linspace(t_start, t_end, t_number_of_images))
    outs = run(params,ts)
    for k in range(len(ts)):
        height, width = img1.shape[:2]
        img_array = outs[k].reshape((height, width, 4))

        img_array = np.nan_to_num(img_array)          # replace NaN/Inf with 0
        img_array = np.clip(img_array, 0, 255)        # clamp values to [0, 255]
        img_array = img_array.astype(np.uint8)        # convert to unsigned 8-bit integer

        # Save output
        pretty_t = format(ts[k], ".2f").replace('.','p')
        output_filename = args.output+"_t"+pretty_t 
        output_filename = output_filename +".png"
        Image.fromarray(img_array).save(output_filename)
        print(f"Saved result to {output_filename}")

# --------------------------------------------------------------------

if __name__ == "__main__":
    main()
