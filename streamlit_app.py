#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
φ-FIELD ADVANCED KNOWLEDGE MINER - STREAMLIT EDITION v12.0
═══════════════════════════════════════════════════════════════════════════════
Interactive Streamlit dashboard for unified knowledge mining and φ-resonance detection

AUTHOR: Peter Braun
DATE: 2025-02-04
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import json
import math
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
SNL_UNIVERSAL_K = 1.05e15

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="φ-Field Knowledge Miner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .phi-constant {
        text-align: center;
        font-size: 1.5rem;
        color: #ffd700;
        padding: 1rem;
        background: rgba(255,215,0,0.1);
        border-radius: 10px;
        margin: 1rem 0;
    }
    .entity-card {
        background: rgba(255,215,0,0.05);
        border-left: 4px solid #ffd700;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MiningEntity:
    id: str
    type: str
    content: str
    source: str
    timestamp: str
    confidence: float
    phi_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINES
# ═══════════════════════════════════════════════════════════════════════════

class EntropyAnalyzer:
    @staticmethod
    def calculate(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        length = len(text)
        entropy = -sum((c / length) * math.log2(c / length) for c in counts.values())
        max_entropy = math.log2(min(256, len(set(text))))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    @staticmethod
    def is_signal(text: str, min_entropy: float = 0.25) -> bool:
        return len(text) >= 20 and EntropyAnalyzer.calculate(text) > min_entropy

class PatternDetector:
    PATTERNS = {
        'equation': re.compile(
            r'\$\$([^\$]+)\$\$|\$([^\$]+)\$|([A-Z][a-z]*(?:_[a-z]+)*)\s*=\s*([^=\n]{5,100})',
            re.DOTALL
        ),
        'phi_ref': re.compile(r'φ|phi|golden\s+ratio|1\.618|fibonacci', re.IGNORECASE),
        'evidence': re.compile(r'r\s*=\s*(-?0\.\d+)|p\s*[<>=]\s*(0\.\d+)', re.IGNORECASE),
        'prediction': re.compile(r'predict(?:s|ion)?[\s:]+([^.\n]{10,200})', re.IGNORECASE)
    }
    
    @staticmethod
    def extract_equations(text: str) -> List[str]:
        equations = []
        for match in PatternDetector.PATTERNS['equation'].finditer(text):
            eq = next((g for g in match.groups() if g), None)
            if eq and len(eq.strip()) > 3:
                equations.append(eq.strip())
        return equations
    
    @staticmethod
    def detect_phi_scaling(text: str) -> int:
        return len(PatternDetector.PATTERNS['phi_ref'].findall(text))

class PhiScoreCalculator:
    @staticmethod
    def calculate_phi_score(entropy: float, frequency: int, depth: int) -> float:
        return (entropy * math.log1p(frequency)) * (0.9 ** depth) / PHI
    
    @staticmethod
    def calculate_phi_resonance(score1: float, score2: float) -> Tuple[float, float, str]:
        if score2 == 0:
            return 0.0, 1.0, "NONE"
        ratio = score1 / score2
        deviation = abs(ratio - PHI) / PHI
        
        if deviation < 0.1:
            strength = "STRONG"
        elif deviation < 0.3:
            strength = "MODERATE"
        elif deviation < 0.5:
            strength = "WEAK"
        else:
            strength = "NONE"
        return ratio, deviation, strength

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

if 'entities' not in st.session_state:
    st.session_state.entities = {}
if 'phi_resonances' not in st.session_state:
    st.session_state.phi_resonances = []
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'files_processed': 0,
        'entities_mined': 0,
        'phi_resonances': 0,
        'equations_found': 0,
        'predictions_found': 0
    }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-header">🔬 φ-Field Advanced Knowledge Miner</div>', unsafe_allow_html=True)
st.markdown(f'<div class="phi-constant">φ = {PHI:.10f} | K = {SNL_UNIVERSAL_K:.2e} s·rad²·m⁻³</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ Mining Configuration")
    
    st.markdown("### Upload Data")
    uploaded_files = st.file_uploader(
        "Upload JSON/TXT files",
        type=['json', 'txt', 'md'],
        accept_multiple_files=True
    )
    
    st.markdown("---")
    st.markdown("### Detection Parameters")
    
    phi_threshold = st.slider("φ-Resonance Threshold", 0.0, 1.0, 0.7, 0.05)
    entropy_min = st.slider("Minimum Entropy", 0.0, 1.0, 0.25, 0.05)
    max_display = st.slider("Max Entities Display", 10, 500, 100, 10)
    
    st.markdown("---")
    st.markdown("### Pattern Detection")
    
    enable_equations = st.checkbox("Extract Equations", value=True)
    enable_phi = st.checkbox("Detect φ-Scaling", value=True)
    enable_predictions = st.checkbox("Extract Predictions", value=True)
    
    st.markdown("---")
    
    if st.button("🔍 Start Mining", type="primary"):
        if uploaded_files:
            progress_bar = st.progress(0)
            status = st.empty()
            
            for idx, file in enumerate(files):
                status.text(f"Processing: {file.name}")
                
                try:
                    content = file.read().decode('utf-8')
                    
                    entropy = EntropyAnalyzer.calculate(content)
                    
                    if EntropyAnalyzer.is_signal(content, entropy_min):
                        phi_count = PatternDetector.detect_phi_scaling(content) if enable_phi else 0
                        frequency = content.count(' ') + 1
                        phi_score = PhiScoreCalculator.calculate_phi_score(entropy, frequency, 0)
                        confidence = min(1.0, entropy + 0.1 * phi_count)
                        
                        equations = PatternDetector.extract_equations(content) if enable_equations else []
                        st.session_state.stats['equations_found'] += len(equations)
                        
                        entity_id = hashlib.md5((file.name + content[:100]).encode()).hexdigest()[:12]
                        
                        entity = MiningEntity(
                            id=entity_id,
                            type='document',
                            content=content[:1000],
                            source=file.name,
                            timestamp=datetime.now().isoformat(),
                            confidence=confidence,
                            phi_score=phi_score,
                            metadata={
                                'entropy': entropy,
                                'phi_references': phi_count,
                                'equations': equations,
                                'length': len(content)
                            }
                        )
                        
                        st.session_state.entities[entity_id] = entity
                        st.session_state.stats['entities_mined'] += 1
                    
                    st.session_state.stats['files_processed'] += 1
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # Detect resonances
            entities = list(st.session_state.entities.values())
            for i in range(len(entities)):
                for j in range(i+1, min(i+50, len(entities))):
                    e1, e2 = entities[i], entities[j]
                    
                    if e1.phi_score > 0 and e2.phi_score > 0:
                        ratio, deviation, strength = PhiScoreCalculator.calculate_phi_resonance(
                            e1.phi_score, e2.phi_score
                        )
                        
                        if strength in ['STRONG', 'MODERATE']:
                            st.session_state.phi_resonances.append({
                                'entity1': e1.id,
                                'entity2': e2.id,
                                'ratio': ratio,
                                'deviation': deviation,
                                'strength': strength
                            })
                            st.session_state.stats['phi_resonances'] += 1
            
            status.text("✅ Complete!")
            progress_bar.empty()
            st.success(f"Extracted {st.session_state.stats['entities_mined']} entities!")
        else:
            st.warning("Upload files first!")
    
    if st.button("🔄 Clear All"):
        st.session_state.entities = {}
        st.session_state.phi_resonances = []
        st.session_state.stats = {k: 0 for k in st.session_state.stats}
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.entities:
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📁 Files", st.session_state.stats['files_processed'])
    with col2:
        st.metric("📦 Entities", st.session_state.stats['entities_mined'])
    with col3:
        st.metric("✨ φ-Resonances", st.session_state.stats['phi_resonances'])
    with col4:
        st.metric("📐 Equations", st.session_state.stats['equations_found'])
    with col5:
        avg_phi = sum(e.phi_score for e in st.session_state.entities.values()) / len(st.session_state.entities)
        st.metric("Avg φ-Score", f"{avg_phi:.4f}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🕸️ Knowledge Graph", "🔬 Entity Analysis"])
    
    with tab1:
        st.header("Mining Overview")
        
        entities_list = list(st.session_state.entities.values())
        phi_scores = [e.phi_score for e in entities_list]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                x=phi_scores,
                nbins=30,
                title='φ-Score Distribution',
                labels={'x': 'φ-Score', 'y': 'Count'}
            )
            fig.update_traces(marker_color='gold')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            confidences = [e.confidence for e in entities_list]
            fig2 = px.histogram(
                x=confidences,
                nbins=30,
                title='Confidence Distribution',
                labels={'x': 'Confidence', 'y': 'Count'}
            )
            fig2.update_traces(marker_color='#667eea')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Top entities
        st.subheader("🌟 Top Entities by φ-Score")
        
        top = sorted(entities_list, key=lambda e: e.phi_score, reverse=True)[:10]
        
        for entity in top:
            with st.expander(f"**{entity.source}** - φ: {entity.phi_score:.4f}"):
                st.write(f"**Confidence:** {entity.confidence:.4f}")
                st.write(f"**Entropy:** {entity.metadata.get('entropy', 0):.4f}")
                
                if entity.metadata.get('equations'):
                    st.write("**Equations:**")
                    for eq in entity.metadata['equations'][:3]:
                        st.code(eq)
                
                st.text(entity.content[:300] + "...")
    
    with tab2:
        st.header("Knowledge Graph")
        
        # Build graph
        G = nx.Graph()
        
        for eid, entity in st.session_state.entities.items():
            G.add_node(eid, phi_score=entity.phi_score, source=entity.source)
        
        for res in st.session_state.phi_resonances:
            G.add_edge(res['entity1'], res['entity2'], weight=1.0-res['deviation'])
        
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            edge_trace = go.Scatter(
                x=[], y=[],
                line=dict(width=0.5, color='#ffd700'),
                mode='lines',
                opacity=0.3
            )
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])
            
            node_trace = go.Scatter(
                x=[], y=[],
                text=[],
                mode='markers',
                marker=dict(
                    size=[],
                    color=[],
                    colorscale='Viridis',
                    showscale=True,
                    line=dict(width=2, color='gold')
                )
            )
            
            for node in G.nodes():
                x, y = pos[node]
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])
                
                entity = st.session_state.entities[node]
                node_trace['marker']['size'] += tuple([10 + entity.phi_score * 30])
                node_trace['marker']['color'] += tuple([entity.phi_score])
                node_trace['text'] += tuple([entity.source])
            
            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                title='φ-Resonance Network',
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload files to build graph")
    
    with tab3:
        st.header("Entity Analysis")
        
        sort_by = st.selectbox("Sort by", ['φ-Score', 'Confidence', 'Entropy'])
        
        filtered = list(st.session_state.entities.values())
        
        if sort_by == 'φ-Score':
            filtered.sort(key=lambda e: e.phi_score, reverse=True)
        elif sort_by == 'Confidence':
            filtered.sort(key=lambda e: e.confidence, reverse=True)
        else:
            filtered.sort(key=lambda e: e.metadata.get('entropy', 0), reverse=True)
        
        for entity in filtered[:max_display]:
            st.markdown(f"""
            <div class="entity-card">
                <strong>{entity.source}</strong><br>
                φ: {entity.phi_score:.4f} | Conf: {entity.confidence:.4f} | H: {entity.metadata.get('entropy', 0):.4f}
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Details"):
                st.json(entity.metadata)
                st.text(entity.content)

else:
    st.info("""
    ### 👋 Welcome!
    
    1. Upload JSON/TXT files in the sidebar
    2. Configure detection parameters
    3. Click "Start Mining"
    4. Explore results in the tabs
    """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p><strong>φ-Field Advanced Knowledge Miner v12.0</strong></p>
    <p>φ = {PHI:.10f}</p>
</div>
""", unsafe_allow_html=True)
```

---

## **requirements.txt**
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
networkx>=3.1
