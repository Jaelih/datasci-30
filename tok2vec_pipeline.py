"""
Tok2Vec-based NER Pipeline for Tagalog

This module provides a unified interface for training BiLSTM, BiLSTM-CRF, and CRF models
with static word embeddings (tok2vec) for Named Entity Recognition.

Models supported:
- bilstm: BiLSTM with softmax output (no CRF)
- bilstm-crf: BiLSTM with CRF layer (custom component)
- crf: CRF-only model (custom component)

All models use spaCy's Tok2Vec with MultiHashEmbed + MaxoutWindowEncoder for embeddings.

Usage:
    # Generate config
    python tok2vec_pipeline.py --action create-config --model bilstm-crf --output configs/bilstm_crf_tok2vec.cfg --gpu-id 0
    
    # Train BiLSTM model (without CRF)
    python tok2vec_pipeline.py --action train --model bilstm --train corpus/train.spacy --dev corpus/dev.spacy --output training/bilstm_tok2vec --gpu-id 0
    
    # Train BiLSTM-CRF model
    python tok2vec_pipeline.py --action train --model bilstm-crf --train corpus/train.spacy --dev corpus/dev.spacy --output training/bilstm_crf_tok2vec --gpu-id 0
    
    # Evaluate model
    python tok2vec_pipeline.py --action evaluate --model-path training/bilstm_crf_tok2vec/model-best --test corpus/test.spacy --output metrics/bilstm_crf_tok2vec.json --gpu-id 0
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List
from wasabi import msg

# Try importing PyTorch dependencies
try:
    import torch
    import torch.nn as nn
    from torchcrf import CRF
    import numpy
    from spacy.language import Language
    from spacy.pipeline import TrainablePipe
    from spacy.tokens import Doc
    from thinc.api import Model, PyTorchWrapper
    from spacy.training import offsets_to_biluo_tags, biluo_tags_to_offsets
    from spacy import util
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    msg.fail(f"PyTorch not available: {e}")
    msg.text("BiLSTM-CRF training requires: torch, torchcrf, spacy, thinc")
    sys.exit(1)


# ============================================================================
# CUSTOM BiLSTM-CRF COMPONENTS
# ============================================================================

class BiLSTM(nn.Module):
    """BiLSTM model for sequence labeling (without CRF)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int, dropout: float = 0.3):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=1,
            batch_first=True, 
            bidirectional=True,
            dropout=0.0
        )
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
    
    def forward(self, embeddings, labels=None, mask=None, return_confidence=False):
        """
        Forward pass through BiLSTM.
        
        Args:
            embeddings: (batch_size, seq_len, input_dim)
            labels: (batch_size, seq_len) - optional, for training
            mask: (batch_size, seq_len) - boolean mask for valid tokens
            return_confidence: If True, return confidence scores alongside predictions
            
        Returns:
            If training (labels provided): cross-entropy loss
            If inference without confidence: predicted tag indices
            If inference with confidence: (predicted tag indices, confidence scores, full probs)
        """
        # BiLSTM forward pass
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        
        # Compute logits
        logits = self.hidden2tag(lstm_out)  # (batch_size, seq_len, num_labels)
        
        # Training mode: compute cross-entropy loss
        if labels is not None:
            labels = torch.clamp(labels, 0, self.num_labels - 1)
            
            # Flatten for loss computation
            logits_flat = logits.view(-1, self.num_labels)
            labels_flat = labels.view(-1)
            
            # Apply mask if provided
            if mask is not None:
                mask_flat = mask.view(-1)
                logits_flat = logits_flat[mask_flat]
                labels_flat = labels_flat[mask_flat]
            
            loss = nn.functional.cross_entropy(logits_flat, labels_flat)
            return loss
        
        # Inference mode: return predictions and optionally confidence scores
        predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
        
        if return_confidence:
            # Apply softmax to get probabilities for all labels
            probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, num_labels)
            # Get confidence for predicted labels (max probability)
            confidence = torch.gather(probs, 2, predictions.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)
            return predictions, confidence, probs  # Return predictions, confidences, and full probability distribution
        
        return predictions.tolist()


class CRFOnly(nn.Module):
    """CRF-only model for sequence labeling (no BiLSTM)."""
    
    def __init__(self, input_dim: int, num_labels: int, dropout: float = 0.3):
        super(CRFOnly, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = 0  # No hidden layer for CRF-only
        self.num_labels = num_labels
        
        self.dropout = nn.Dropout(dropout)
        
        # Emission layer (directly from embeddings)
        self.hidden2tag = nn.Linear(input_dim, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, embeddings, labels=None, mask=None, return_confidence=False):
        """
        Forward pass through CRF only.
        
        Args:
            embeddings: (batch_size, seq_len, input_dim)
            labels: (batch_size, seq_len) - optional, for training
            mask: (batch_size, seq_len) - boolean mask for valid tokens
            return_confidence: If True, return emissions for CRF marginal computation
            
        Returns:
            If training (labels provided): negative log-likelihood loss
            If inference without confidence: decoded tag sequences
            If inference with confidence: (decoded sequences, emissions, mask)
        """
        # Apply dropout to embeddings
        embeddings = self.dropout(embeddings)
        
        # Compute emissions directly from embeddings
        emissions = self.hidden2tag(embeddings)
        
        # Training mode: compute CRF loss
        if labels is not None:
            labels = torch.clamp(labels, 0, self.num_labels - 1)
            if mask is None:
                mask = torch.ones(labels.shape, dtype=torch.bool, device=labels.device)
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        
        # Inference mode: decode best path
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        decoded = self.crf.decode(emissions, mask=mask)
        
        if return_confidence:
            # Return decoded sequences, emissions (for marginals), and mask
            return decoded, emissions, mask
        
        return decoded


class BiLSTMCRF(nn.Module):
    """BiLSTM-CRF model for sequence labeling."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int, dropout: float = 0.3):
        super(BiLSTMCRF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=1,
            batch_first=True, 
            bidirectional=True,
            dropout=0.0  # No dropout for single layer
        )
        self.dropout = nn.Dropout(dropout)
        
        # Emission layer
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, embeddings, labels=None, mask=None, return_confidence=False):
        """
        Forward pass through BiLSTM and CRF.
        
        Args:
            embeddings: (batch_size, seq_len, input_dim)
            labels: (batch_size, seq_len) - optional, for training
            mask: (batch_size, seq_len) - boolean mask for valid tokens
            return_confidence: If True, return BiLSTM confidences and emissions for CRF marginals
            
        Returns:
            If training (labels provided): negative log-likelihood loss
            If inference without confidence: decoded tag sequences
            If inference with confidence: (decoded sequences, bilstm_confidence, emissions, mask)
        """
        # BiLSTM forward pass
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        
        # Compute emissions
        emissions = self.hidden2tag(lstm_out)  # (batch_size, seq_len, num_labels)
        
        # Training mode: compute CRF loss
        if labels is not None:
            labels = torch.clamp(labels, 0, self.num_labels - 1)
            if mask is None:
                mask = torch.ones(labels.shape, dtype=torch.bool, device=labels.device)
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        
        # Inference mode: decode best path
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        decoded = self.crf.decode(emissions, mask=mask)
        
        if return_confidence:
            # Compute BiLSTM softmax probabilities (before CRF)
            bilstm_probs = torch.softmax(emissions, dim=-1)  # (batch_size, seq_len, num_labels)
            # Get BiLSTM confidence for decoded tags
            decoded_tensor = torch.tensor(decoded, device=emissions.device)  # (batch_size, seq_len)
            bilstm_confidence = torch.gather(bilstm_probs, 2, decoded_tensor.unsqueeze(-1)).squeeze(-1)
            
            # Return decoded, BiLSTM confidence, emissions (for CRF marginals), and mask
            # CRF marginals will be computed separately in predict()
            return decoded, bilstm_confidence, emissions, mask
        
        return decoded


def compute_crf_marginals(crf_layer, emissions, mask):
    """
    Compute marginal probabilities for each token position using forward-backward algorithm.
    
    Args:
        crf_layer: The CRF layer with transitions
        emissions: (batch_size, seq_len, num_labels) - emission scores
        mask: (batch_size, seq_len) - boolean mask for valid tokens
        
    Returns:
        marginals: (batch_size, seq_len, num_labels) - marginal probability for each label at each position
    """
    batch_size, seq_len, num_labels = emissions.shape
    device = emissions.device
    
    # Get transition scores from CRF
    transitions = crf_layer.transitions  # (num_labels, num_labels)
    start_transitions = crf_layer.start_transitions  # (num_labels,)
    end_transitions = crf_layer.end_transitions  # (num_labels,)
    
    # Forward pass: compute alpha (forward variables)
    # alpha[t, j] = log sum of scores of all paths that end at state j at time t
    alpha = torch.zeros(batch_size, seq_len, num_labels, device=device)
    
    # Initialize with start transitions + first emissions
    alpha[:, 0, :] = start_transitions + emissions[:, 0, :]
    
    # Forward pass for remaining positions
    for t in range(1, seq_len):
        # For each current state j
        for j in range(num_labels):
            # Sum over all previous states i: alpha[t-1, i] + transition[i, j] + emission[t, j]
            # Use log-sum-exp for numerical stability
            scores = alpha[:, t-1, :] + transitions[:, j].unsqueeze(0) + emissions[:, t, j].unsqueeze(1)
            alpha[:, t, j] = torch.logsumexp(scores, dim=1)
        
        # Apply mask
        if mask is not None:
            alpha[:, t, :] = torch.where(
                mask[:, t].unsqueeze(1),
                alpha[:, t, :],
                alpha[:, t-1, :]
            )
    
    # Backward pass: compute beta (backward variables)
    # beta[t, i] = log sum of scores of all paths from state i at time t to the end
    beta = torch.zeros(batch_size, seq_len, num_labels, device=device)
    
    # Initialize with end transitions
    for b in range(batch_size):
        seq_end = seq_len - 1
        if mask is not None:
            seq_end = mask[b].sum().item() - 1
        beta[b, seq_end, :] = end_transitions
    
    # Backward pass for remaining positions
    for t in range(seq_len - 2, -1, -1):
        # For each current state i
        for i in range(num_labels):
            # Sum over all next states j: beta[t+1, j] + transition[i, j] + emission[t+1, j]
            scores = beta[:, t+1, :] + transitions[i, :].unsqueeze(0) + emissions[:, t+1, :]
            beta[:, t, i] = torch.logsumexp(scores, dim=1)
        
        # Apply mask
        if mask is not None:
            beta[:, t, :] = torch.where(
                mask[:, t].unsqueeze(1),
                beta[:, t, :],
                beta[:, t+1, :]
            )
    
    # Compute partition function Z (log of sum of all path scores)
    # Z = log sum exp(alpha[T-1, i] + end_transitions[i]) for all i
    final_scores = alpha[:, -1, :] + end_transitions.unsqueeze(0)
    Z = torch.logsumexp(final_scores, dim=1, keepdim=True)  # (batch_size, 1)
    
    # Compute marginals: P(y_t = j | x) = exp(alpha[t, j] + beta[t, j] - Z)
    marginals = torch.exp(alpha + beta - Z.unsqueeze(2))  # (batch_size, seq_len, num_labels)
    
    return marginals


class BiLSTMCRFNER(TrainablePipe):
    """
    spaCy pipeline component for BiLSTM/BiLSTM-CRF NER.
    Uses spaCy's built-in BILUO encoding utilities.
    """
    
    def __init__(self, vocab, model, name="bilstm_crf_ner", *, tok2vec=None, transformer=None, labels=None, use_crf=True):
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = {}
        self.tok2vec = tok2vec
        self.transformer = transformer
        self.use_crf = use_crf  # Flag to distinguish BiLSTM vs BiLSTM-CRF
        
        # BILUO label mapping
        self._label_map = {}
        self._idx_to_label = {}
        self._repair_count = 0
        self._max_repair_logs = 5

        # Determine model type based on the actual PyTorch model class
        pytorch_model = self.model.attrs.get("pytorch_model")
        if pytorch_model is not None:
            model_class_name = pytorch_model.__class__.__name__
            if model_class_name == "BiLSTMCRF":
                model_type = "bilstm-crf"
            elif model_class_name == "BiLSTM":
                model_type = "bilstm"
            elif model_class_name == "CRFOnly":
                model_type = "crf"
            else:
                model_type = model_class_name
        else:
            # Fallback to name-based detection
            if "bilstm" in name.lower() and "crf" in name.lower():
                model_type = "bilstm-crf"
            elif "bilstm" in name.lower():
                model_type = "bilstm"
            elif "crf" in name.lower():
                model_type = "crf"
            else:
                model_type = name
        
        msg.info(f"[model type] {model_type}")
        
        if labels:
            self._initialize_labels(labels)
    
    def _initialize_labels(self, labels):
        """Initialize BILUO label mappings from entity types."""
        biluo_labels = ["O"]
        
        for label in sorted(labels):
            biluo_labels.extend([
                f"B-{label}", f"I-{label}", f"L-{label}", f"U-{label}"
            ])
        
        self._label_map = {label: idx for idx, label in enumerate(biluo_labels)}
        self._idx_to_label = {idx: label for label, idx in self._label_map.items()}
        
        msg.info(f"Initialized {len(biluo_labels)} BILUO labels for {len(labels)} entity types")
    
    @property
    def labels(self):
        """Return entity type labels (not BILUO tags)."""
        entity_labels = set()
        for label in self._label_map.keys():
            if label != "O" and "-" in label:
                entity_labels.add(label.split("-", 1)[1])
        return tuple(sorted(entity_labels))
    
    def _spans_to_biluo(self, doc):
        """Convert spaCy spans to BILUO tags."""
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        biluo_tags = offsets_to_biluo_tags(doc, entities)
        return [self._label_map.get(tag, 0) for tag in biluo_tags]
    
    def _biluo_to_spans(self, doc, tag_indices):
        """Convert BILUO tag indices to spaCy spans."""
        biluo_tags = [self._idx_to_label.get(idx, "O") for idx in tag_indices]
        
        # Repair invalid BILUO sequences
        def _repair_biluo(tags: List[str]) -> List[str]:
            repaired = list(tags)
            for i, tag in enumerate(repaired):
                if tag == "O" or "-" not in tag:
                    continue
                prefix, label = tag.split("-", 1)
                if prefix == "I":
                    if i == 0 or not repaired[i-1].endswith(f"-{label}") or not repaired[i-1].startswith(("B-", "I-")):
                        repaired[i] = f"B-{label}"
                elif prefix == "L":
                    if i == 0 or not repaired[i-1].endswith(f"-{label}") or not repaired[i-1].startswith(("B-", "I-")):
                        repaired[i] = f"U-{label}"
            return repaired
        
        repaired_tags = _repair_biluo(biluo_tags)
        entities = biluo_tags_to_offsets(doc, repaired_tags)
        
        spans = []
        for start_char, end_char, label in entities:
            span = doc.char_span(start_char, end_char, label=label)
            if span is not None:
                spans.append((span.start, span.end, label))
        
        return spans
    
    def predict(self, docs):
        """Predict entities for a batch of documents."""
        if not docs:
            return []
        
        pytorch_model = self.model.attrs["pytorch_model"]
        pytorch_model.eval()
        device = next(pytorch_model.parameters()).device
        model_class_name = pytorch_model.__class__.__name__
        
        # Get embeddings from tok2vec
        if self.tok2vec is not None:
            tokvecs_output = self.tok2vec.predict(docs)
            tokvecs = []
            for tokvec_array in tokvecs_output:
                if hasattr(tokvec_array, 'get'):
                    tokvec_np = tokvec_array.get()
                else:
                    tokvec_np = numpy.asarray(tokvec_array)
                tokvecs.append(tokvec_np)
        else:
            raise ValueError("tok2vec must be available")
        
        predictions = []
        confidence_data = []
        
        # Check if we should compute confidence (set via extract_confidence_mode attribute)
        need_confidence = getattr(self, 'extract_confidence_mode', False)
        
        with torch.no_grad():
            for doc, tokvec in zip(docs, tokvecs):
                seq_len = min(len(doc), len(tokvec))
                if seq_len == 0:
                    predictions.append([])
                    if need_confidence:
                        confidence_data.append({
                            'bilstm_confidence': [],
                            'crf_confidence': [],
                            'bilstm_probs': [],
                            'crf_marginals': [],
                            'model_type': model_class_name
                        })
                    continue
                
                tokvec = tokvec[:seq_len]
                embeddings = torch.tensor(tokvec, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Only compute confidence if needed
                output = pytorch_model(embeddings, return_confidence=need_confidence)
                
                if need_confidence:
                    doc_confidence = {'model_type': model_class_name}
                    
                    if model_class_name == "BiLSTM":
                        decoded, confidence, probs = output
                        if isinstance(decoded, torch.Tensor):
                            decoded_list = decoded[0].cpu().numpy().tolist() if decoded.dim() > 1 else decoded.cpu().numpy().tolist()
                        else:
                            decoded_list = decoded[0] if decoded and len(decoded) > 0 else [0] * seq_len
                        predictions.append(decoded_list)
                        
                        doc_confidence['bilstm_confidence'] = confidence[0].cpu().numpy().tolist() if confidence is not None else []
                        doc_confidence['bilstm_probs'] = probs[0].cpu().numpy().tolist() if probs is not None else []
                        doc_confidence['crf_confidence'] = []
                        doc_confidence['crf_marginals'] = []
                        
                    elif model_class_name == "BiLSTMCRF":
                        decoded, bilstm_confidence, emissions, mask = output
                        predictions.append(decoded[0] if decoded and len(decoded) > 0 else [0] * seq_len)
                        
                        doc_confidence['bilstm_confidence'] = bilstm_confidence[0].cpu().numpy().tolist() if bilstm_confidence is not None else []
                        bilstm_probs = torch.softmax(emissions, dim=-1) if emissions is not None else None
                        doc_confidence['bilstm_probs'] = bilstm_probs[0].cpu().numpy().tolist() if bilstm_probs is not None else []
                        
                        crf_marginals = compute_crf_marginals(pytorch_model.crf, emissions, mask)
                        doc_confidence['crf_marginals'] = crf_marginals[0].cpu().numpy().tolist()
                        
                        crf_confidence_list = []
                        for t, pred_label in enumerate(decoded[0]):
                            if t < crf_marginals.shape[1]:
                                crf_confidence_list.append(crf_marginals[0, t, pred_label].item())
                        doc_confidence['crf_confidence'] = crf_confidence_list
                        
                    elif model_class_name == "CRFOnly":
                        decoded, emissions, mask = output
                        predictions.append(decoded[0] if decoded and len(decoded) > 0 else [0] * seq_len)
                        
                        doc_confidence['bilstm_confidence'] = []
                        doc_confidence['bilstm_probs'] = []
                        
                        crf_marginals = compute_crf_marginals(pytorch_model.crf, emissions, mask)
                        doc_confidence['crf_marginals'] = crf_marginals[0].cpu().numpy().tolist()
                        
                        crf_confidence_list = []
                        for t, pred_label in enumerate(decoded[0]):
                            if t < crf_marginals.shape[1]:
                                crf_confidence_list.append(crf_marginals[0, t, pred_label].item())
                        doc_confidence['crf_confidence'] = crf_confidence_list
                    
                    confidence_data.append(doc_confidence)
                else:
                    # Fast path: no confidence computation
                    if isinstance(output, tuple):
                        decoded = output[0]
                    else:
                        decoded = output
                    
                    if isinstance(decoded, torch.Tensor):
                        decoded_list = decoded[0].cpu().numpy().tolist() if decoded.dim() > 1 else decoded.cpu().numpy().tolist()
                    elif isinstance(decoded, list) and len(decoded) > 0:
                        decoded_list = decoded[0] if isinstance(decoded[0], list) else decoded
                    else:
                        decoded_list = [0] * seq_len
                    
                    predictions.append(decoded_list)
        
        # Store confidence data only if computed
        if need_confidence:
            if not Doc.has_extension("confidence"):
                Doc.set_extension("confidence", default=None)
            
            for doc, conf_data in zip(docs, confidence_data):
                doc._.confidence = conf_data
        
        return predictions
    
    def set_annotations(self, docs, predictions):
        """Set entity annotations on documents."""
        for doc, tags in zip(docs, predictions):
            spans = self._biluo_to_spans(doc, tags)
            ents = []
            for start, end, label in spans:
                span = doc[start:end]
                span.label_ = label
                ents.append(span)
            
            try:
                doc.ents = ents
            except ValueError:
                doc.ents = util.filter_spans(ents)
    
    def score(self, examples, **kwargs):
        """Score a batch of examples."""
        from spacy.scorer import Scorer
        scorer = Scorer()
        return scorer.score_spans(examples, "ents", **kwargs)
    
    def update(self, examples, *, drop=0.0, sgd=None, losses=None):
        """Update the model on a batch of examples."""
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        
        if not examples:
            return losses
        
        pytorch_model = self.model.attrs["pytorch_model"]
        device = next(pytorch_model.parameters()).device
        pytorch_model.train()
        
        if not hasattr(self, '_pytorch_optimizer'):
            self._pytorch_optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)
        
        docs = [eg.predicted for eg in examples]
        gold_docs = [eg.reference for eg in examples]
        
        # Get embeddings from tok2vec
        if self.tok2vec is None:
            raise ValueError("tok2vec must be available")
        
        tokvecs, bp_tokvecs = self.tok2vec.begin_update(docs)
        
        # Convert to numpy - handle both single array and list of arrays
        if hasattr(tokvecs, '__iter__') and not hasattr(tokvecs, 'shape'):
            # It's a list of arrays
            tokvecs_np = []
            for tokvec in tokvecs:
                if hasattr(tokvec, 'get'):
                    tokvecs_np.append(tokvec.get())
                else:
                    tokvecs_np.append(numpy.asarray(tokvec))
        else:
            # It's a single array, convert to numpy
            if hasattr(tokvecs, 'get'):
                tokvecs_np = tokvecs.get()
            else:
                tokvecs_np = tokvecs
        
        tokvec_gradients = []
        self._pytorch_optimizer.zero_grad()
        
        total_loss = 0.0
        num_valid_docs = 0
        
        for doc_idx, (doc, tokvec, gold_doc) in enumerate(zip(docs, tokvecs_np, gold_docs)):
            gold_biluo = self._spans_to_biluo(gold_doc)
            seq_len = min(len(doc), len(tokvec), len(gold_biluo))
            
            if seq_len == 0:
                tokvec_gradients.append(numpy.zeros_like(tokvec))
                continue
            
            gold_biluo = gold_biluo[:seq_len]
            tokvec = tokvec[:seq_len]
            
            if isinstance(tokvec, numpy.ndarray):
                if tokvec.ndim == 3:
                    tokvec = numpy.squeeze(tokvec, axis=0)
                embeddings = torch.tensor(tokvec, dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
            else:
                embeddings = torch.tensor(tokvec, dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
            
            embeddings.retain_grad()
            
            labels = torch.tensor(gold_biluo, dtype=torch.long, device=device).unsqueeze(0)
            labels = torch.clamp(labels, 0, len(self._label_map) - 1)
            
            loss = pytorch_model(embeddings, labels)
            loss.backward()
            
            if embeddings.grad is not None:
                grad = embeddings.grad.squeeze(0).cpu().numpy()
                full_grad = numpy.zeros_like(tokvecs_np[doc_idx])
                full_grad[:seq_len] = grad
                tokvec_gradients.append(full_grad)
            else:
                tokvec_gradients.append(numpy.zeros_like(tokvecs_np[doc_idx]))
            
            total_loss += loss.item()
            num_valid_docs += 1
        
        if num_valid_docs > 0:
            self._pytorch_optimizer.step()
            losses[self.name] += total_loss / num_valid_docs
        
        if tokvec_gradients and num_valid_docs > 0 and bp_tokvecs is not None:
            # Convert gradients to the same array type as the input (CuPy for GPU, numpy for CPU)
            try:
                import cupy
                # If we have CuPy and the model is on GPU, convert to CuPy arrays
                if device.type == "cuda":
                    tokvec_gradients = [cupy.asarray(grad) for grad in tokvec_gradients]
            except ImportError:
                # CuPy not available, keep as numpy arrays
                pass
            
            bp_tokvecs(tokvec_gradients)
            if sgd is not None:
                self.tok2vec.finish_update(sgd)
        
        return losses
    
    def initialize(self, get_examples, *, nlp=None, labels=None):
        """Initialize the model and labels."""
        # Check if static vectors are available (after vectors are loaded during initialization)
        if self.vocab.vectors.shape[0] > 0:
            msg.info(f"[static vectors] enabled - shape: {self.vocab.vectors.shape} (name: {self.vocab.vectors.name})")
        else:
            msg.info("[static vectors] disabled - no vectors loaded")
        
        if labels is not None:
            self._initialize_labels(labels)
        else:
            entity_labels = set()
            for example in get_examples():
                for ent in example.reference.ents:
                    entity_labels.add(ent.label_)
            self._initialize_labels(sorted(entity_labels))
        
        if nlp is not None and "tok2vec" in nlp.pipe_names:
            tok2vec_component = nlp.get_pipe("tok2vec")
            self.tok2vec = tok2vec_component.model  # Store the model, not the component
        
        # Recreate model with correct num_labels
        num_labels = len(self._label_map)
        if num_labels > 0:
            old_pytorch_model = self.model.attrs.get("pytorch_model")
            if old_pytorch_model:
                input_dim = old_pytorch_model.input_dim
                hidden_dim = old_pytorch_model.hidden_dim
                dropout = old_pytorch_model.dropout.p if hasattr(old_pytorch_model, 'dropout') else 0.3
                
                # Create the correct model type
                if self.use_crf:
                    if hidden_dim == 0:
                        # CRF-only model
                        new_pytorch_model = CRFOnly(input_dim, num_labels, dropout)
                    else:
                        # BiLSTM-CRF model
                        new_pytorch_model = BiLSTMCRF(input_dim, hidden_dim, num_labels, dropout)
                else:
                    # BiLSTM-only model
                    new_pytorch_model = BiLSTM(input_dim, hidden_dim, num_labels, dropout)
                
                self.model.attrs["pytorch_model"] = new_pytorch_model
                
                device = next(old_pytorch_model.parameters()).device
                new_pytorch_model.to(device)
                
                if self.use_crf and hidden_dim == 0:
                    model_type = "CRF-only"
                elif self.use_crf:
                    model_type = "BiLSTM-CRF"
                else:
                    model_type = "BiLSTM"
                msg.good(f"{model_type} model recreated with {num_labels} labels")
    
    def to_disk(self, path, *, exclude=tuple()):
        """Serialize the component to disk."""
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        
        with (path / "labels.json").open("w") as f:
            json.dump({
                "label_map": self._label_map,
                "idx_to_label": self._idx_to_label
            }, f, indent=2)
        
        pytorch_model = self.model.attrs["pytorch_model"]
        torch.save(pytorch_model.state_dict(), path / "model.pt")
        
        if hasattr(self, '_pytorch_optimizer'):
            torch.save(self._pytorch_optimizer.state_dict(), path / "optimizer.pt")
    
    def from_disk(self, path, *, exclude=tuple()):
        """Load the component from disk."""
        path = Path(path)
        
        with (path / "labels.json").open("r") as f:
            label_data = json.load(f)
            self._label_map = label_data["label_map"]
            self._idx_to_label = {int(k): v for k, v in label_data["idx_to_label"].items()}
        
        num_labels = len(self._label_map)
        old_pytorch_model = self.model.attrs["pytorch_model"]
        
        input_dim = old_pytorch_model.input_dim
        hidden_dim = old_pytorch_model.hidden_dim
        dropout = old_pytorch_model.dropout.p if hasattr(old_pytorch_model, 'dropout') else 0.3
        device = next(old_pytorch_model.parameters()).device
        
        # Create the correct model type
        if self.use_crf:
            if hidden_dim == 0:
                # CRF-only model
                pytorch_model = CRFOnly(input_dim, num_labels, dropout)
            else:
                # BiLSTM-CRF model
                pytorch_model = BiLSTMCRF(input_dim, hidden_dim, num_labels, dropout)
        else:
            # BiLSTM-only model
            pytorch_model = BiLSTM(input_dim, hidden_dim, num_labels, dropout)
        
        pytorch_model.to(device)
        self.model.attrs["pytorch_model"] = pytorch_model
        pytorch_model.load_state_dict(torch.load(path / "model.pt"))
        
        if (path / "optimizer.pt").exists() and hasattr(self, '_pytorch_optimizer'):
            self._pytorch_optimizer.load_state_dict(torch.load(path / "optimizer.pt"))
        
        return self


def build_bilstm_model(input_dim, hidden_dim=256, dropout=0.3, num_labels=13):
    """Build a PyTorchWrapper model for BiLSTM (without CRF)."""
    pytorch_model = BiLSTM(input_dim, hidden_dim, num_labels, dropout)
    model = PyTorchWrapper(pytorch_model)
    model.attrs["pytorch_model"] = pytorch_model
    return model


def build_bilstm_crf_model(input_dim, hidden_dim=256, dropout=0.3, num_labels=13):
    """Build a PyTorchWrapper model for BiLSTM-CRF or CRF-only."""
    if hidden_dim == 0:
        # CRF-only model (no BiLSTM)
        pytorch_model = CRFOnly(input_dim, num_labels, dropout)
    else:
        # BiLSTM-CRF model
        pytorch_model = BiLSTMCRF(input_dim, hidden_dim, num_labels, dropout)
    model = PyTorchWrapper(pytorch_model)
    model.attrs["pytorch_model"] = pytorch_model
    return model


@Language.factory(
    "bilstm",
    default_config={
        "hidden_dim": 256,
        "dropout": 0.3,
        "labels": None
    }
)
def make_bilstm_ner(nlp, name, hidden_dim, dropout, labels):
    """Factory function to create BiLSTM NER component (without CRF)."""
    tok2vec_model = None
    input_dim = 256
    
    if "tok2vec" in nlp.pipe_names:
        tok2vec_component = nlp.get_pipe("tok2vec")
        tok2vec_model = tok2vec_component.model  # Get the model from the component
        try:
            input_dim = tok2vec_model.get_dim("nO")
        except:
            input_dim = 256
    
    entity_labels = labels if labels else []
    num_labels = 1 + (len(entity_labels) * 4) if entity_labels else 1
    
    model = build_bilstm_model(input_dim, hidden_dim, dropout, num_labels)
    
    return BiLSTMCRFNER(nlp.vocab, model, name=name, tok2vec=tok2vec_model, transformer=None, labels=entity_labels, use_crf=False)


@Language.factory(
    "bilstm-crf",
    default_config={
        "hidden_dim": 256,
        "dropout": 0.3,
        "labels": None
    }
)
def make_bilstm_crf_ner(nlp, name, hidden_dim, dropout, labels):
    """Factory function to create BiLSTM-CRF NER component."""
    tok2vec_model = None
    input_dim = 256
    
    if "tok2vec" in nlp.pipe_names:
        tok2vec_component = nlp.get_pipe("tok2vec")
        tok2vec_model = tok2vec_component.model  # Get the model from the component
        try:
            input_dim = tok2vec_model.get_dim("nO")
        except:
            input_dim = 256
    
    entity_labels = labels if labels else []
    num_labels = 1 + (len(entity_labels) * 4) if entity_labels else 1
    
    model = build_bilstm_crf_model(input_dim, hidden_dim, dropout, num_labels)
    
    return BiLSTMCRFNER(nlp.vocab, model, name=name, tok2vec=tok2vec_model, transformer=None, labels=entity_labels, use_crf=True)


@Language.factory(
    "crf",
    default_config={
        "dropout": 0.3,
        "labels": None
    }
)
def make_crf_ner(nlp, name, dropout, hidden_dim, labels): # only include hidden_dim for compatibility. It will not be use here.
    """Factory function to create CRF-only NER component (no BiLSTM)."""
    tok2vec_model = None
    input_dim = 256
    
    if "tok2vec" in nlp.pipe_names:
        tok2vec_component = nlp.get_pipe("tok2vec")
        tok2vec_model = tok2vec_component.model  # Get the model from the component
        try:
            input_dim = tok2vec_model.get_dim("nO")
        except:
            input_dim = 256
    
    entity_labels = labels if labels else []
    num_labels = 1 + (len(entity_labels) * 4) if entity_labels else 1
    
    # Use hidden_dim=0 to create CRF-only model
    model = build_bilstm_crf_model(input_dim, hidden_dim=0, dropout=dropout, num_labels=num_labels)
    
    return BiLSTMCRFNER(nlp.vocab, model, name=name, tok2vec=tok2vec_model, transformer=None, labels=entity_labels, use_crf=True)


def create_config(
    model_type: str,
    lang: str = "tl",
    use_gpu: bool = False,
    hidden_dim: int = 256,
    dropout: float = 0.3
) -> str:
    """
    Create a spaCy config for tok2vec-based NER models.
    
    Args:
        model_type: One of 'bilstm', 'bilstm-crf', 'crf'
        lang: Language code
        use_gpu: Whether to use GPU
        hidden_dim: Hidden dimension for BiLSTM/CRF
        dropout: Dropout rate
        
    Returns:
        Configuration string
    """
    gpu_allocator = "pytorch" if use_gpu else "null"
    
    # Base config with tok2vec
    config = f"""
[paths]
train = null
dev = null
vectors = null
init_tok2vec = null

[system]
gpu_allocator = {gpu_allocator}
seed = 0

[nlp]
lang = "{lang}"
pipeline = ["tok2vec","{model_type.replace('-', '_')}_ner"]
batch_size = 1000
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {{"@tokenizers":"spacy.Tokenizer.v1"}}

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = ${{components.tok2vec.model.encode.width}}
attrs = ["NORM","PREFIX","SUFFIX","SHAPE"]
rows = [5000,1000,2500,2500]
include_static_vectors = true

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 256
depth = 8
window_size = 1
maxout_pieces = 3

[components.{model_type.replace('-', '_')}_ner]
factory = "{model_type.replace('-', '_')}_ner"
hidden_dim = {hidden_dim}
dropout = {dropout}

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${{system.seed}}
gpu_allocator = ${{system.gpu_allocator}}
dropout = 0.1
accumulate_gradient = 1
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 5
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001
learn_rate = 0.001

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0
ents_per_type = null

[pretraining]

[initialize]
vectors = ${{paths.vectors}}
init_tok2vec = ${{paths.init_tok2vec}}
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
"""
    if model_type not in ("bilstm", "bilstm-crf", "crf"):
        raise ValueError(f"Unknown model type: {model_type}. Choose from: bilstm, bilstm-crf, crf")
    
    return config


def train_model(
    model_type: str,
    train_path: str,
    dev_path: str,
    output_dir: str,
    gpu_id: int = -1,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    vectors: Optional[str] = None,
    init_tok2vec: Optional[str] = None
):
    """
    Train a tok2vec-based NER model.
    
    Args:
        model_type: One of 'bilstm', 'bilstm-crf', 'crf'
        train_path: Path to training data
        dev_path: Path to development data
        output_dir: Output directory for trained model
        gpu_id: GPU ID to use (-1 for CPU)
        hidden_dim: Hidden dimension
        dropout: Dropout rate
        vectors: Optional path to pretrained word vectors
        init_tok2vec: Optional path to pretrained tok2vec model for initialization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate config
    msg.info(f"Generating config for {model_type} model...")
    use_gpu = gpu_id >= 0
    if model_type == "crf":
        hidden_dim = 0  # crf model does not need hidden dim
    config_str = create_config(model_type, use_gpu=use_gpu, hidden_dim=hidden_dim, dropout=dropout)
    
    # Save config
    config_path = output_path / "config.cfg"
    with open(config_path, "w") as f:
        f.write(config_str)
    msg.good(f"Config saved to {config_path}")
    
    # Build training command
    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(config_path),
        "--output", str(output_path),
        "--paths.train", train_path,
        "--paths.dev", dev_path,
        "--code", __file__  # Import custom components from this file
    ]
    
    if gpu_id >= 0:
        cmd.extend(["--gpu-id", str(gpu_id)])
    
    if vectors:
        cmd.extend(["--paths.vectors", vectors])
        msg.info(f"Using pretrained vectors from: {vectors}")
    
    if init_tok2vec:
        cmd.extend(["--paths.init_tok2vec", init_tok2vec])
        msg.info(f"Initializing tok2vec from: {init_tok2vec}")
    
    msg.info(f"Training {model_type} model...")
    msg.text(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        msg.good(f"Training completed successfully!")
        msg.text(f"Model saved to: {output_path / 'model-best'}")
    except subprocess.CalledProcessError as e:
        msg.fail(f"Training failed with exit code {e.returncode}")
        sys.exit(1)


def evaluate_model(
    model_path: str,
    test_path: str,
    gpu_id: int = -1,
    output_path: Optional[str] = None
):
    """
    Evaluate a trained NER model.
    
    Args:
        model_path: Path to trained model
        test_path: Path to test data
        gpu_id: GPU ID to use (-1 for CPU)
        output_path: Optional path to save metrics JSON
    """
    cmd = [
        sys.executable, "-m", "spacy", "evaluate",
        model_path,
        test_path,
        "--code", __file__
    ]
    
    if gpu_id >= 0:
        cmd.extend(["--gpu-id", str(gpu_id)])
    
    if output_path:
        cmd.extend(["--output", output_path])
    
    msg.info(f"Evaluating model: {model_path}")
    msg.text(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        msg.good("Evaluation completed successfully!")
        if output_path:
            msg.text(f"Metrics saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        msg.fail(f"Evaluation failed with exit code {e.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Tok2Vec-based NER Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train BiLSTM-CRF with tok2vec
  python tok2vec_pipeline.py --action train --model bilstm-crf --train corpus/train.spacy --dev corpus/dev.spacy --output training/bilstm_crf_tok2vec --gpu-id 0
  
  # Train with pretrained word vectors
  python tok2vec_pipeline.py --action train --model bilstm-crf --train corpus/train.spacy --dev corpus/dev.spacy --output training/bilstm_crf_tok2vec --vectors path/to/vectors --gpu-id 0
  
  # Train with pretrained tok2vec initialization
  python tok2vec_pipeline.py --action train --model bilstm-crf --train corpus/train.spacy --dev corpus/dev.spacy --output training/bilstm_crf_tok2vec --init-tok2vec training/baseline/model-best --gpu-id 0
  
  # Train with both vectors and tok2vec initialization
  python tok2vec_pipeline.py --action train --model bilstm-crf --train corpus/train.spacy --dev corpus/dev.spacy --output training/bilstm_crf_tok2vec --vectors path/to/vectors --init-tok2vec training/baseline/model-best --gpu-id 0
  
  # Evaluate model
  python tok2vec_pipeline.py --action evaluate --model-path training/bilstm_crf_tok2vec/model-best --test corpus/test.spacy --output metrics/bilstm_crf_tok2vec.json
  
  # Generate config only
  python tok2vec_pipeline.py --action create-config --model bilstm-crf --output configs/bilstm_crf_tok2vec.cfg
        """
    )
    
    parser.add_argument(
        "--action",
        choices=["train", "evaluate", "create-config"],
        required=True,
        help="Action to perform"
    )
    
    parser.add_argument(
        "--model",
        choices=["bilstm", "bilstm-crf", "crf"],
        help="Model type (required for train and create-config)"
    )
    
    parser.add_argument("--train", help="Path to training data")
    parser.add_argument("--dev", help="Path to development data")
    parser.add_argument("--test", help="Path to test data")
    parser.add_argument("--output", help="Output directory or file path")
    parser.add_argument("--model-path", help="Path to trained model (for evaluation)")
    parser.add_argument("--gpu-id", type=int, default=-1, help="GPU ID (-1 for CPU)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for BiLSTM")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--vectors", help="Path to pretrained word vectors")
    parser.add_argument("--init-tok2vec", help="Path to pretrained tok2vec model for initialization")
    
    args = parser.parse_args()
    
    if args.action == "train":
        if not args.model or not args.train or not args.dev or not args.output:
            msg.fail("--model, --train, --dev, and --output are required for training")
            sys.exit(1)
        
        train_model(
            model_type=args.model,
            train_path=args.train,
            dev_path=args.dev,
            output_dir=args.output,
            gpu_id=args.gpu_id,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            vectors=args.vectors,
            init_tok2vec=args.init_tok2vec
        )
    
    elif args.action == "evaluate":
        if not args.model_path or not args.test:
            msg.fail("--model-path and --test are required for evaluation")
            sys.exit(1)
        
        evaluate_model(
            model_path=args.model_path,
            test_path=args.test,
            gpu_id=args.gpu_id,
            output_path=args.output
        )
    
    elif args.action == "create-config":
        if not args.model or not args.output:
            msg.fail("--model and --output are required for config creation")
            sys.exit(1)
        
        use_gpu = args.gpu_id >= 0
        config_str = create_config(
            model_type=args.model,
            use_gpu=use_gpu,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(config_str)
        
        msg.good(f"Config saved to {output_path}")


if __name__ == "__main__":
    main()
