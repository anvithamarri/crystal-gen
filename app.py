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
st.set_page_config(page_title="StructGen",layout="wide",initial_sidebar_state="expanded")
# --- UI Configuration & Styling ---
# Custom CSS for better organization
st.markdown("""
    <style>
        .header-title {
            text-align: center;
            color: #00d4ff;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 0.5em;
            letter-spacing: 2px;
        }
        .subtitle {
            text-align: center;
            color: #888;
            font-size: 1.1em;
            margin-bottom: 2em;
        }
        .section-header {
            color: #00d4ff;
            border-bottom: 2px solid #00d4ff;
            padding-bottom: 0.5em;
            margin-top: 1.5em;
        }
    </style>
""", unsafe_allow_html=True)

# --- Main Header ---
st.markdown("<div class='header-title'>StructGen</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Crystal Structure Generation & Optimization</div>", unsafe_allow_html=True)

# Initialize
# Initialize
with st.spinner("Initializing AI & CHGNet..."):
    model, tokenizer, device, status_msg = load_backend()

# --- Sidebar: System Status ---
with st.sidebar:
    st.markdown("<h2 style='color: #00d4ff;'>⚙️ System Configuration</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Hardware", device.upper())
    with col2:
        st.metric("Status", "Ready" if status_msg == "Success" else " Error")
    
    if status_msg != "Success":
        st.error(f"Backend Error: {status_msg}")
    
    st.divider()
    
    # --- Search Settings (Collapsible) ---
    with st.expander(" Search Settings", expanded=True):
        num_sims = st.slider("Number of Simulations", 5, 500, 150)
        width = st.slider("Tree Width (Top-K)", 2, 50, 20) 
        temp = st.slider("Temperature", 0.1, 1.5, 0.1)
    
    st.divider()
    
    # --- Physical Constraints (Collapsible) ---
    with st.expander(" Physical Constraints", expanded=True):
        target_rho = st.slider("Target Density (g/cm³)", 1.0, 15.0, 3.51)
        c_puct = st.number_input("Exploration Weight (PUCT)", value=1.4, step=0.1)
# Main Interface
# --- Main Content Area ---
st.markdown("<div class='section-header'>Input Configuration</div>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    formula = st.text_input(
        "Chemical Formula",
        value="C 2",
        placeholder="e.g., C 2, NaCl, Fe3O4"
    )
with col2:
    run_button = st.button(" Run Optimization", type="primary", use_container_width=True)

# --- Main Optimization Process ---
if run_button:
    from scorer import CHGNetScorer
    from model_utils import GPTConfig
    from mcts import MCTSSampler, MCTSEvaluator, PUCTSelector, ContextSensitiveTreeBuilder

    clean_formula = formula.replace(" ", "")
    start_prompt = f"data_{clean_formula}\n"
    
    try:
        external_scorer = CHGNetScorer()
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
            
            # --- Post-Processing Options ---
            st.markdown("<div class='section-header'> Post-Processing Options</div>", unsafe_allow_html=True)
            
            with st.expander(" Structural Relaxation (CHGNet - Ceder Group AI)", expanded=False):
                st.write("Apply CHGNet to refine atomic positions using machine learning force field")
                do_relax = st.checkbox("Enable CHGNet Post-Optimization", value=False)
                
                if do_relax:
                    with st.spinner(" CHGNet is pulling atoms into stable positions..."):
                        cif_output = relax_structure(cif_output, device)
                        st.success(" Structure successfully relaxed by CHGNet!")
            
            # --- Results Display ---
            st.markdown("<div class='section-header'> Optimization Results</div>", unsafe_allow_html=True)
            
            st.info(f" Generation Successful! | Physical Score: **{best_score:.4f}**")
            
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                st.markdown("####  3D Crystal Lattice")
                try:
                    view = py3Dmol.view(width=700, height=500)
                    view.addModel(cif_output, 'cif')
                    view.setStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}, 
                                   'stick': {'colorscheme': 'Jmol', 'radius': 0.1}})
                    view.addUnitCell()
                    view.replicateUnitCell(2, 2, 2)
                    view.setBackgroundColor('#121212')
                    view.zoomTo()
                    st.components.v1.html(view._make_html(), height=500)
                except Exception as ve:
                    st.error(" 3D Viewer Error - Unable to display structure")

            with col2:
                st.markdown("####  Structure Data")
                
                # Download button
                st.download_button(
                    label=" Download CIF File",
                    data=cif_output,
                    file_name=f"{clean_formula}_opt.cif",
                    mime="text/plain",
                    use_container_width=True
                )
                
                # Raw CIF Display
                with st.expander(" Show Raw CIF Text", expanded=False):
                    st.code(cif_output, language="text")
                
                # Summary Stats
                st.markdown("#####  Metadata")
                st.metric("Formula", clean_formula)
                st.metric("Score", f"{best_score:.4f}")
            
           
        else:
            st.error("No valid structure found. Try increasing 'Width' to 25.")

    except Exception:
        st.error("The optimization process crashed.")
        with st.expander("Error Details", expanded=False):
        st.code(traceback.format_exc())
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888; margin-top: 3em;'>
        <p>StructGen • AI-Powered Crystal Structure Generation</p>
        <p style='font-size: 0.8em;'>Powered by CrystalLM & CHGNet</p>
    </div>
    """,
    unsafe_allow_html=True
)
