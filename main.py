import streamlit as st
import sys
import os
import logging
from pathlib import Path
import numpy as np
import argparse
import llama_cpp
from llama_cpp import llama_model_quantize_params
from argparse import Namespace
import subprocess
from huggingface_hub import snapshot_download
from pathlib import Path
import time
import threading
import queue

# set the python path to include the llama.cpp directory
sys.path.append(os.path.join(os.getcwd(), "llama.cpp"))

uploaded_file = None

DEFAULT_CONCURRENCY = int(os.getenv("DEFAULT_CONCURRENCY", 2))

script_path = "llama.cpp/convert_hf_to_gguf.py"
# Define the enum
ggml_type_enum = {
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 8,
    "Q5_1": 9,
    "IQ2_XXS": 19,
    "IQ2_XS": 20,
    "IQ2_S": 28,
    "IQ2_M": 29,
    "IQ1_S": 24,
    "IQ1_M": 31,
    "Q2_K": 10,
    "Q2_K_S": 21,
    "IQ3_XXS": 23,
    "IQ3_S": 26,
    "IQ3_M": 27,
    "Q3_K": 12,
    "IQ3_XS": 22,
    "Q3_K_S": 11,
    "Q3_K_M": 12,
    "Q3_K_L": 13,
    "IQ4_NL": 25,
    "IQ4_XS": 30,
    "Q4_K": 15,
    "Q4_K_S": 14,
    "Q4_K_M": 15,
    "Q5_K": 17,
    "Q5_K_S": 16,
    "Q5_K_M": 17,
    "Q6_K": 18,
    "Q8_0": 7,
    "F16": 1,
    "BF16": 32,
    "F32": 0,
    "COPY": -1
}

ggml_type_enum_invert = {v: k for k, v in ggml_type_enum.items()}


def is_huggingface_repo_id(input_str):
    """Check if input looks like a HuggingFace repo ID rather than a local path."""
    input_str = input_str.strip()
    
    if input_str.startswith('/') or '\\' in input_str:
        return False
    
    if ':' in input_str and len(input_str.split(':')) == 2 and len(input_str.split(':')[0]) == 1:
        return False
    
    if Path(input_str).exists():
        return False
    
    return True


def run_subprocess_with_live_output(cmd, description="Running command"):
    """Run subprocess and show live output in streamlit."""
    output_placeholder = st.empty()
    output_lines = []
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            errors='replace'
        )
        
        # Read output line by line
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # Show last 20 lines
                display_lines = output_lines[-20:] if len(output_lines) > 20 else output_lines
                output_placeholder.text_area(
                    f"{description} - Live Output:",
                    value='\n'.join(display_lines),
                    height=200
                )
        
        # Get the return code
        return_code = process.poll()
        return return_code, '\n'.join(output_lines)
        
    except Exception as e:
        return -1, str(e)


def streamlit_main():
    st.title("GGUF_GUI")
    st.header("Step 1: Convert Safetensor to GGUF")
    output_choices = (
        ["f32", "f16", "bf16", "q8_0", "auto"]
        if np.uint32(1) == np.uint32(1).newbyteorder("<")
        else ["f32", "f16"]
    )

    # Create interface elements
    manual_entry_str = st.text_input("Enter Directory Path or repo", placeholder="/path/to/safetensors/ or username_or_org/repo_name")
    root_output_path = Path(st.text_input("Enter path to save your work to.", placeholder="/path/to/files/"))
    outtype = st.selectbox("Output format", output_choices, index=0)
    awq_path = st.text_input("AWQ path (optional)")
    verbose = st.checkbox("Verbose logging")
    outfile_input = st.text_input("Output file name (optional)", "")
    vocab_only = st.checkbox("Extract only the vocab")
    big_endian = st.checkbox("Model is executed on big endian machine")
    
    # Conversion mode selection
    conversion_mode = st.radio(
        "Output mode:",
        ["Capture output (safe)", "Live output (real-time)"],
        index=0
    )
    
    # Button to trigger the main function
    if st.button("Run Conversion"):
        if not manual_entry_str.strip():
            st.error("Please enter a directory path or repo ID")
            return
        
        if not root_output_path or not str(root_output_path).strip():
            st.error("Please enter an output path")
            return
        
        try:
            with st.spinner(f"Converting Safetensors to {outtype}"):
                # Determine if input is a local path or HuggingFace repo ID
                if is_huggingface_repo_id(manual_entry_str):
                    st.info(f"Detected HuggingFace repo ID: {manual_entry_str}")
                    repo_name = manual_entry_str.split('/')[-1]
                    local_download_path = root_output_path / repo_name
                    
                    try:
                        safetensor_dl_loc = snapshot_download(
                            repo_id=manual_entry_str, 
                            local_files_only=False, 
                            local_dir=str(local_download_path)
                        )
                        st.success(f"Model downloaded to {safetensor_dl_loc}")
                    except Exception as e:
                        st.error(f"Failed to download from HuggingFace: {e}")
                        return
                else:
                    manual_entry_path = Path(manual_entry_str)
                    if not manual_entry_path.exists():
                        st.error(f"Local directory does not exist: {manual_entry_str}")
                        return
                    
                    if not manual_entry_path.is_dir():
                        st.error(f"Path is not a directory: {manual_entry_str}")
                        return
                    
                    model_files = list(manual_entry_path.glob("*.safetensors")) + \
                                 list(manual_entry_path.glob("*.bin")) + \
                                 list(manual_entry_path.glob("config.json"))
                    
                    if not model_files:
                        st.error(f"No model files found in directory: {manual_entry_str}")
                        return
                    
                    safetensor_dl_loc = str(manual_entry_path)
                    st.success(f"Using local directory: {safetensor_dl_loc}")
                
                # Generate output filename
                if is_huggingface_repo_id(manual_entry_str):
                    model_name = manual_entry_str.split('/')[-1]
                else:
                    model_name = Path(manual_entry_str).name
                
                outfile = f"{root_output_path}/{model_name}_{outtype}.gguf"
                if vocab_only:
                    outfile = outfile.replace(".gguf", "_vocab_only.gguf")
                if big_endian:
                    outfile = outfile.replace(".gguf", "_big_endian.gguf")
                
                st.session_state['st_to_gguf_outfile'] = outfile

                # Prepare arguments
                args = ["--outfile", outfile, "--outtype", outtype]
                
                if verbose:
                    args.append("--verbose")
                if vocab_only:
                    args.append("--vocab-only")
                if big_endian:
                    args.append("--bigendian")
                if awq_path:
                    args.extend(["--awq-path", awq_path])
                
                args.append(safetensor_dl_loc)
                
                st.info(f"Running conversion command: python {script_path} {' '.join(args)}")
                
                # Execute based on selected mode
                if conversion_mode == "Live output (real-time)":
                    return_code, output = run_subprocess_with_live_output(
                        ["python", script_path] + args,
                        "Conversion"
                    )
                else:
                    result = subprocess.run(
                        ["python", script_path] + args, 
                        capture_output=True, 
                        text=True, 
                        errors='replace'
                    )
                    return_code = result.returncode
                    output = result.stdout + result.stderr
                
                if return_code == 0:
                    st.success("Conversion completed successfully!")
                    st.success(f"Wrote file to: {outfile}")
                else:
                    st.error(f"Conversion failed with return code {return_code}")
                    if output:
                        st.error(f"Output: {output}")
                        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
    
    # Step 2: Quantization
    st.header("Step 2: Quantize FP32/16")
    
    ggml_selected_type = st.selectbox("Select a Quantization Type", list(ggml_type_enum.keys()))
    outfile_ggml = st.text_input("Enter file path to your FP 32/16 GGUF", st.session_state.get('st_to_gguf_outfile', ''), placeholder="/path/to/input/model.gguf")
    
    # Quantization mode selection
    quant_mode = st.radio(
        "Quantization output mode:",
        ["Capture output (safe)", "Live output (real-time)"],
        index=0,
        key="quant_mode"
    )
    
    with st.expander("Optional Parameters", expanded=False):
        model_quant = st.text_input("Quantized Model File Path (optional)", "")
        allow_requantize = st.checkbox("Allow Requantize")
        leave_output_tensor = st.checkbox("Leave Output Tensor")
        pure = st.checkbox("Pure")
        imatrix_file = st.text_input("Importance Matrix File", "")
        include_weights = st.text_input("Include Weights Tensor", "")
        exclude_weights = st.text_input("Exclude Weights Tensor", "")
        output_tensor_type = st.text_input("Output Tensor Type", "")
        token_embedding_type = st.text_input("Token Embedding Type", "")
        override_kv = st.text_area("Override KV", "")
        nthreads = st.number_input("Number of Threads", min_value=1, step=1, value=DEFAULT_CONCURRENCY)

    if include_weights and exclude_weights:
        st.error("Cannot use --include-weights and --exclude-weights together.")
        return
    
    imatrix = st.checkbox("iMatrix")

    if imatrix:
        with st.expander("iMatrix Parameters", expanded=True):
            training_data = st.text_input("Training Data File Path (-f)", "")
            imatrix_output_file = st.text_input("Output File Path (-o)", str(root_output_path / "imatrix.dat") if root_output_path else "imatrix.dat")
            verbosity_level = st.selectbox("Verbosity Level", [None, "0", "1", "2", "3"], index=0)
            num_chunks = st.number_input("Number of Chunks (-ofreq)", min_value=1, step=1, value=10)
            ow_option = st.selectbox("Overwrite Option (-ow)", [None, "0", "1"], index=0)
            other_params = st.text_area("Other Parameters", "", placeholder="--arg value --arg2 value2")

    if st.button("Quantize"):
        if not outfile_ggml:
            st.warning("Please enter an input GGUF file path.")
            return
            
        if not Path(outfile_ggml).exists():
            st.error(f"Input GGUF file does not exist: {outfile_ggml}")
            return
            
        with st.spinner(f"Converting to {ggml_selected_type}"):
            try:
                # iMatrix step
                if imatrix:
                    if not training_data:
                        st.error("Training data file is required for iMatrix")
                        return
                        
                    cmd = ["./llama.cpp/build/bin/llama-imatrix"]
                    cmd.extend(["-m", outfile_ggml, "-f", training_data, "-o", imatrix_output_file])
                    
                    if verbosity_level:
                        cmd.extend(["--verbosity", verbosity_level])
                    if num_chunks:
                        cmd.extend(["-ofreq", str(num_chunks)])
                    if ow_option:
                        cmd.extend(["-ow", ow_option])
                    if other_params:
                        cmd.extend(other_params.split())
                    
                    st.info(f"Running iMatrix: {' '.join(cmd)}")
                    
                    if quant_mode == "Live output (real-time)":
                        return_code, output = run_subprocess_with_live_output(cmd, "iMatrix")
                    else:
                        result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
                        return_code = result.returncode
                        output = result.stdout + result.stderr
                    
                    if return_code != 0:
                        st.error(f"iMatrix failed: {output}")
                        return
                    else:
                        st.success("iMatrix completed successfully!")

                # Quantization step
                cmd = ["./llama.cpp/build/bin/llama-quantize"]
                
                if imatrix and imatrix_output_file:
                    cmd.extend(["--imatrix", imatrix_output_file])
                elif imatrix_file:
                    cmd.extend(["--imatrix", imatrix_file])
                    
                if allow_requantize:
                    cmd.append("--allow-requantize")
                if leave_output_tensor:
                    cmd.append("--leave-output-tensor")
                if pure:
                    cmd.append("--pure")
                if include_weights:
                    cmd.extend(["--include-weights", include_weights])
                if exclude_weights:
                    cmd.extend(["--exclude-weights", exclude_weights])
                if output_tensor_type:
                    cmd.extend(["--output-tensor-type", output_tensor_type])
                if token_embedding_type:
                    cmd.extend(["--token-embedding-type", token_embedding_type])
                if override_kv:
                    for kv in override_kv.split('\n'):
                        if kv.strip():
                            cmd.extend(["--override-kv", kv.strip()])

                infile_ggml = outfile_ggml
                outfile_ggml_quantized = infile_ggml.replace(".gguf", f"_{ggml_selected_type}.gguf")

                cmd.extend([infile_ggml, outfile_ggml_quantized, str(ggml_type_enum[ggml_selected_type])])
                
                if nthreads:
                    cmd.append(str(nthreads))

                st.info(f"Running quantization: {' '.join(cmd)}")
                
                # Run quantization
                if quant_mode == "Live output (real-time)":
                    return_code, output = run_subprocess_with_live_output(cmd, "Quantization")
                else:
                    result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
                    return_code = result.returncode
                    output = result.stdout + result.stderr
                
                if return_code == 0:
                    st.success("Quantization completed successfully!")
                    st.success(f"Output quantized model to: {outfile_ggml_quantized}")
                else:
                    st.error(f"Quantization failed with return code {return_code}")
                    if output:
                        st.error(f"Output: {output}")
                        
            except Exception as e:
                st.error(f"An error occurred during quantization: {e}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    streamlit_main()
