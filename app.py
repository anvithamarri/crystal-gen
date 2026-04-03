import streamlit as st
import torch
import os
import traceback
import numpy as np
import py3Dmol
from ase.io import read, write
import io
from ase.optimize import BFGS
# OFFICIAL CHGNet IMPORT
from chgnet.model.dynamics import CHGNetCalculator 

# --- Official CHGNet Relaxation Function ---
def relax_structure(cif_string, device):
    try:
        # Convert bytes to string if necessary
        if isinstance(cif_string, bytes):
            cif_string = cif_string.decode('utf-8')
            
        # Load structure from the model's output
        struct = read(io.StringIO(cif_string), format='cif')
        
        # Initialize the Official CHGNet Calculator
        # It will use CUDA if available, otherwise CPU
        calc = CHGNetCalculator(use_device=device)
        struct.calc = calc
        
        # Use BFGS Optimizer to relax the structure (Ceder Group standard)
        dyn = BFGS(struct, logfile=None) 
        dyn.run(fmax=0.05) # Pull atoms until forces are below 0.05 eV/A
        
        out_buf = io.StringIO()
        write(out_buf, struct, format='cif')
        return out_buf.getvalue()
    except Exception as e:
        st.warning(f"CHGNet Relaxation skipped: {e}")
        return cif_string

# --- Backend Loading (CrystalLM Model) ---
@st.cache_resource
def load_backend():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt_path = os.path.join("models", "crystallm_200m", "ckpt.pt")
    
    if not os.path.exists(ckpt_path):
        return None, None, device, f"Checkpoint not found at: {os.path.abspath(ckpt_path)}"

    try:
        from scorer import HeuristicPhysicalScorer
        from model_utils import GPT, GPTConfig, CIFTokenizer
        from mcts import MCTSSampler, MCTSEvaluator, PUCTSelector, ContextSensitiveTreeBuilder
        
        tokenizer = CIFTokenizer()
        config = GPTConfig()
        model = GPT(config)
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            if k.startswith('_orig_mod.'):
                state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        return model, tokenizer, device, "Success"
    except Exception as e:
        return None, None, device, str(e)

# --- UI Setup ---
st.set_page_config(page_title="Crystal-Gen Pro", layout="wide")
st.markdown("<h1 style='text-align: center; color: #00d4ff;'>Crystal-Gen</h1>", unsafe_allow_html=True)

# Initialize
with st.spinner("Initializing AI & CHGNet..."):
    model, tokenizer, device, status_msg = load_backend()

# Sidebar: Config
st.sidebar.header("System Status")
st.sidebar.success(f"Hardware: {device.upper()}")

st.sidebar.divider()
st.sidebar.header("Search Settings")
num_sims = st.sidebar.slider("Simulations", 5, 500, 150)
width = st.sidebar.slider("Tree Width (Top-K)", 2, 50, 20) 
temp = st.sidebar.slider("Temperature", 0.1, 1.5, 0.1)

st.sidebar.subheader("Physical Constraints")
target_rho = st.sidebar.slider("Target Density (g/cm³)", 1.0, 15.0, 3.51)
c_puct = st.sidebar.number_input("Exploration Weight", value=1.4)

# Main Interface
formula = st.text_input("Chemical Formula", value="C 2")

if st.button("Run Optimization", type="primary"):
    from scorer import HeuristicPhysicalScorer
    from model_utils import GPTConfig
    from mcts import MCTSSampler, MCTSEvaluator, PUCTSelector, ContextSensitiveTreeBuilder

    clean_formula = formula.replace(" ", "")
    start_prompt = f"data_{clean_formula}\n"
    
    try:
        external_scorer = HeuristicPhysicalScorer(target_density=target_rho)
        evaluator = MCTSEvaluator(scorer=external_scorer, tokenizer=tokenizer)
        selector = PUCTSelector(cpuct=c_puct)
        tree_builder = ContextSensitiveTreeBuilder(tokenizer=tokenizer)

        sampler = MCTSSampler(
            model=model,
            config=model.config if hasattr(model, 'config') else GPTConfig(),
            width=width,
            max_depth=512,
            eval_function=evaluator,
            node_selector=selector,
            tokenizer=tokenizer,
            temperature=temp,
            device=device,
            tree_builder=tree_builder
        )

        with st.status(f"MCTS Searching for {clean_formula}...", expanded=True) as status:
            sampler.search(start=start_prompt, num_simulations=num_sims)
            status.update(label="Search Complete!", state="complete", expanded=False)
        
        # Results Handling
        best_data = sampler.get_best_sequence()
        if best_data:
            best_seq, best_score = best_data
            cif_output = tokenizer.decode(best_seq)
            
            st.divider()
            
            # --- OFFICIAL CHGNet STEP ---
            with st.expander("Structural Relaxation (Ceder Group AI)", expanded=True):
                do_relax = st.checkbox("Enable CHGNet Post-Optimization", value=True)
            
            if do_relax:
                with st.spinner("CHGNet is pulling atoms into stable positions..."):
                    cif_output = relax_structure(cif_output, device)
                    st.success("Structure successfully relaxed by CHGNet!")
            
            # Results Layout
            st.info(f"Generation Successful! (Physical Score: {best_score:.4f})")
            col1, col2 = st.columns([1.5, 1])
            with col1:
                st.subheader("3D Crystal Lattice")
                try:
                    view = py3Dmol.view(width=700, height=500)
                    view.addModel(cif_output, 'cif')
                    view.setStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}, 
                                   'stick': {'colorscheme': 'Jmol', 'radius': 0.1}})
                    view.addUnitCell()
                    view.replicateUnitCell(2, 2, 2) # Show supercell for symmetry
                    view.setBackgroundColor('#121212') # Pro Dark Look
                    view.zoomTo()
                    st.components.v1.html(view._make_html(), height=500)
                except Exception as ve:
                    st.error("Viewer error.")

            with col2:
                st.subheader("Structure Data")
                st.download_button("Download CIF", cif_output, f"{clean_formula}_opt.cif")
                with st.expander("Show Raw CIF Text"):
                    st.code(cif_output, language="text")
        else:
            st.error("No valid structure found. Try increasing 'Width' to 25.")

    except Exception:
        st.error("The optimization process crashed.")
        st.code(traceback.format_exc())
