import streamlit as st
import torch
import os
import traceback
import numpy as np

# 1. Imports from your project structure
try:
    # Ensure HeuristicPhysicalScorer is imported
    from scorer import HeuristicPhysicalScorer
    from model_utils import GPT, GPTConfig, CIFTokenizer
    from mcts import MCTSSampler, MCTSEvaluator, PUCTSelector, ContextSensitiveTreeBuilder
except ImportError as e:
    st.error(f"Critical Import Error: {e}")
    st.info("Ensure scorer.py, model_utils.py, and mcts.py are in the same folder.")
    st.stop()

# --- Backend Loading ---
@st.cache_resource
def load_backend():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Path to your checkpoint
    ckpt_path = os.path.join("models", "crystallm_200m", "ckpt.pt")
    
    if not os.path.exists(ckpt_path):
        return None, None, device, f"Checkpoint not found at: {os.path.abspath(ckpt_path)}"

    try:
        tokenizer = CIFTokenizer()
        config = GPTConfig()
        model = GPT(config)
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['model']
        
        # Repair DDP keys if necessary
        for k in list(state_dict.keys()):
            if k.startswith('_orig_mod.'):
                state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        return model, tokenizer, device, "Success"
    except Exception as e:
        return None, None, device, str(e)

# --- UI Setup ---
st.set_page_config(page_title="Crystal-Gen", layout="wide")
st.title("Crystal-Gen")

# Load Backend
with st.spinner("Initializing Model..."):
    model, tokenizer, device, status_msg = load_backend()

# Sidebar Setup
st.sidebar.header("System Status")
if model is None:
    st.sidebar.error(f"Model Load Failed: {status_msg}")
    st.stop()
else:
    st.sidebar.success(f"Running on: {device.upper()}")

# --- MCTS Settings ---
st.sidebar.divider()
st.sidebar.header("Optimization Settings")
# Updated default to 100 for better convergence
num_sims = st.sidebar.slider("Simulations", 5, 500, 100, help="Higher = Better quality but slower")
width = st.sidebar.slider("Tree Width (Top-K)", 2, 50, 5)
# Updated default to 0.5 for more focused generation
temp = st.sidebar.slider("Temperature", 0.1, 1.5, 0.5)

st.sidebar.subheader("Physical Constraints")
# Updated default to 2.16 for NaCl
target_rho = st.sidebar.slider("Target Density (g/cm³)", 1.0, 15.0, 2.16)
c_puct = st.sidebar.number_input("Exploration Weight (C-PUCT)", value=1.4)

# Main Interface
formula = st.text_input("Chemical Formula", value="Na1 Cl1", help="Format: Element1 Count1 Element2 Count2")

if st.button("Run Optimization", type="primary"):
    clean_formula = formula.replace(" ", "")
    start_prompt = f"data_{clean_formula}\n"
    
    try:
        external_scorer = HeuristicPhysicalScorer(target_density=target_rho)
        
        # Setup MCTS components
        evaluator = MCTSEvaluator(scorer=external_scorer, tokenizer=tokenizer)
        selector = PUCTSelector(cpuct=c_puct)
        tree_builder = ContextSensitiveTreeBuilder(tokenizer=tokenizer)

        # Initialize Sampler
        sampler = MCTSSampler(
            model=model,
            config=model.config if hasattr(model, 'config') else GPTConfig(),
            width=width,
            max_depth=1024,
            eval_function=evaluator,
            node_selector=selector,
            tokenizer=tokenizer,
            temperature=temp,
            device=device,
            tree_builder=tree_builder
        )

        # Execution
        with st.status(f"Searching for stable {clean_formula}...", expanded=True) as status:
            st.write("Generating candidates and calculating physical rewards...")
            sampler.search(start=start_prompt, num_simulations=num_sims)
            status.update(label="Search Complete!", state="complete", expanded=False)
        
        # Results Handling
        best_data = sampler.get_best_sequence()
        if best_data:
            best_seq, best_score = best_data
            cif_output = tokenizer.decode(best_seq)
            
            st.divider()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Optimized CIF Structure")
                st.code(cif_output, language="text")
            with col2:
                st.subheader("Actions")
                st.download_button(
                    label="Download Best CIF", 
                    data=cif_output, 
                    file_name=f"{clean_formula}_optimized.cif",
                    mime="text/plain"
                )
        else:
            st.warning("No valid structure was found. Try increasing the number of simulations.")

    except Exception:
        st.error("The optimization process crashed.")
        with st.expander("Show Traceback"):
            st.code(traceback.format_exc())