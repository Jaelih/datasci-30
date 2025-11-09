"""
Extract confidence scores from trained NER models on dev.spacy corpus.
Compares predictions with gold labels and saves detailed confidence information.
"""
import spacy
from spacy.tokens import DocBin, Doc
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from wasabi import msg


def register_factories_for_model(model_path: Path):
    """Register only the factories needed for this specific model."""
    config_path = model_path / "config.cfg"
    
    if not config_path.exists():
        msg.warn(f"Config file not found: {config_path}")
        return
    
    # Read config to determine which factories are needed
    with open(config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    # Check if it's a transformer-based model (has transformer component in pipeline)
    # Transformer models use transformer_components.py
    if '"transformer"' in config_content or 'components.transformer' in config_content:
        import transformer_components
        msg.info("Registered transformer-based factories")
    # Otherwise it's a tok2vec-based model
    else:
        import tok2vec_pipeline
        msg.info("Registered tok2vec-based factories")


def load_corpus(corpus_path: Path) -> List[Doc]:
    """Load spaCy corpus from .spacy file."""
    nlp = spacy.blank("tl")  # Tagalog blank model
    doc_bin = DocBin().from_disk(corpus_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    msg.good(f"Loaded {len(docs)} documents from {corpus_path}")
    return docs


def extract_gold_labels(doc: Doc) -> List[str]:
    """Extract gold standard BILUO labels from document entities."""
    # Initialize all tokens as "O"
    gold_labels = ["O"] * len(doc)
    
    # Convert entities to BILUO tags
    for ent in doc.ents:
        start_token = ent.start
        end_token = ent.end
        label = ent.label_
        
        if end_token - start_token == 1:
            # Single-token entity
            gold_labels[start_token] = f"U-{label}"
        else:
            # Multi-token entity
            gold_labels[start_token] = f"B-{label}"
            for i in range(start_token + 1, end_token - 1):
                gold_labels[i] = f"I-{label}"
            gold_labels[end_token - 1] = f"L-{label}"
    
    return gold_labels


def process_documents(
    nlp: spacy.Language,
    docs: List[Doc],
    model_name: str,
    idx_to_label: Dict[int, str],
    limit: int = None
) -> List[Dict[str, Any]]:
    """
    Process documents through model and extract confidence scores.
    
    Args:
        nlp: Loaded spaCy model
        docs: List of documents to process
        model_name: Name of the model (for metadata)
        idx_to_label: Mapping from label index to label string
        limit: Optional limit on number of documents to process
    
    Returns:
        List of dictionaries containing token-level information
    """
    results = []
    
    # Enable confidence extraction mode on the NER component
    ner = nlp.get_pipe("ner")
    ner.extract_confidence_mode = True
    
    # Limit number of documents if specified
    if limit:
        docs = docs[:limit]
        msg.info(f"Processing first {limit} documents")
    
    for doc_idx, gold_doc in enumerate(docs):
        # Extract gold labels
        gold_labels = extract_gold_labels(gold_doc)
        
        # Create a new doc with just the text (no entities) for prediction
        pred_doc = nlp.make_doc(gold_doc.text)
        
        # Run prediction through pipeline
        try:
            pred_doc = nlp(pred_doc)
        except IndexError as e:
            msg.warn(f"IndexError in document {doc_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
        except Exception as e:
            msg.warn(f"Error processing document {doc_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Get confidence data from custom attribute
        confidence_data = pred_doc._.confidence if hasattr(pred_doc._, "confidence") else None
        
        if confidence_data is None:
            msg.warn(f"No confidence data found for document {doc_idx}. Skipping.")
            continue
        
        # Get predicted entities and convert to BILUO
        pred_labels = ["O"] * len(pred_doc)
        for ent in pred_doc.ents:
            start_token = ent.start
            end_token = ent.end
            label = ent.label_
            
            if end_token - start_token == 1:
                pred_labels[start_token] = f"U-{label}"
            else:
                pred_labels[start_token] = f"B-{label}"
                for i in range(start_token + 1, end_token - 1):
                    pred_labels[i] = f"I-{label}"
                pred_labels[end_token - 1] = f"L-{label}"
        
        # Extract confidence scores and full distributions
        bilstm_confidence = confidence_data.get('bilstm_confidence', [])
        bilstm_probs = confidence_data.get('bilstm_probs', [])  # list of [num_labels] per token
        crf_confidence = confidence_data.get('crf_confidence', [])
        crf_marginals = confidence_data.get('crf_marginals', [])  # list of [num_labels] per token
        model_type = confidence_data.get('model_type', 'unknown')
        
        # Ensure all lists have the same length
        num_tokens = len(pred_doc)
        if len(pred_labels) != num_tokens or len(gold_labels) != num_tokens:
            msg.warn(f"Token count mismatch in doc {doc_idx}. Skipping.")
            continue
        
        # Pad confidence lists if needed
        if len(bilstm_confidence) < num_tokens:
            bilstm_confidence.extend([None] * (num_tokens - len(bilstm_confidence)))
        if len(crf_confidence) < num_tokens:
            crf_confidence.extend([None] * (num_tokens - len(crf_confidence)))
        
        # Create token-level records
        for token_idx, token in enumerate(pred_doc):
            gold_label = gold_labels[token_idx] if token_idx < len(gold_labels) else "O"
            pred_label = pred_labels[token_idx]
            
            # Extract entity type and boundary position
            entity_type = "O"
            boundary_pos = "O"
            if pred_label != "O":
                parts = pred_label.split("-", 1)
                if len(parts) == 2:
                    boundary_pos, entity_type = parts
            
            gold_entity_type = "O"
            gold_boundary_pos = "O"
            if gold_label != "O":
                parts = gold_label.split("-", 1)
                if len(parts) == 2:
                    gold_boundary_pos, gold_entity_type = parts
            
            record = {
                'doc_id': doc_idx,
                'token_position': token_idx,
                'token_text': token.text,
                'gold_label': gold_label,
                'predicted_label': pred_label,
                'correct': gold_label == pred_label,
                'gold_entity_type': gold_entity_type,
                'predicted_entity_type': entity_type,
                'gold_boundary_position': gold_boundary_pos,
                'predicted_boundary_position': boundary_pos,
                'bilstm_confidence': bilstm_confidence[token_idx] if token_idx < len(bilstm_confidence) else None,
                'crf_confidence': crf_confidence[token_idx] if token_idx < len(crf_confidence) else None,
                # Per-label probability distributions (label_str -> prob)
                'bilstm_probs_by_label': {},
                'crf_marginals_by_label': {},
                'model_name': model_name,
                'model_type': model_type
            }
            
            # Fill per-label probabilities if available
            if token_idx < len(bilstm_probs) and bilstm_probs[token_idx]:
                probs_list = bilstm_probs[token_idx]
                try:
                    record['bilstm_probs_by_label'] = {idx_to_label.get(i, str(i)): float(probs_list[i]) for i in range(len(probs_list))}
                except Exception:
                    record['bilstm_probs_by_label'] = {str(i): float(probs_list[i]) for i in range(len(probs_list))}

            if token_idx < len(crf_marginals) and crf_marginals[token_idx]:
                marg_list = crf_marginals[token_idx]
                try:
                    record['crf_marginals_by_label'] = {idx_to_label.get(i, str(i)): float(marg_list[i]) for i in range(len(marg_list))}
                except Exception:
                    record['crf_marginals_by_label'] = {str(i): float(marg_list[i]) for i in range(len(marg_list))}
            
            results.append(record)
    
    msg.good(f"Extracted {len(results)} token records from {len(docs)} documents")
    return results


def save_results(results: List[Dict[str, Any]], output_path: Path, format: str = "json"):
    """Save results to JSON or CSV file."""
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        msg.good(f"Saved results to {output_path}")
    
    elif format == "csv":
        if not results:
            msg.warn("No results to save")
            return
        
        fieldnames = results[0].keys()
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        msg.good(f"Saved results to {output_path}")
    
    else:
        msg.fail(f"Unsupported format: {format}")


def print_summary(results: List[Dict[str, Any]]):
    """Print summary statistics of extracted confidence scores."""
    if not results:
        msg.warn("No results to summarize")
        return
    
    total_tokens = len(results)
    correct_predictions = sum(1 for r in results if r['correct'])
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
    
    # Calculate confidence statistics
    bilstm_confidences = [r['bilstm_confidence'] for r in results if r['bilstm_confidence'] is not None]
    crf_confidences = [r['crf_confidence'] for r in results if r['crf_confidence'] is not None]
    
    msg.divider("Summary Statistics")
    msg.info(f"Total tokens: {total_tokens}")
    msg.info(f"Correct predictions: {correct_predictions} ({accuracy:.2%})")
    msg.info(f"Incorrect predictions: {total_tokens - correct_predictions}")
    
    if bilstm_confidences:
        msg.info(f"\nBiLSTM Confidence:")
        msg.info(f"  Mean: {np.mean(bilstm_confidences):.4f}")
        msg.info(f"  Std: {np.std(bilstm_confidences):.4f}")
        msg.info(f"  Min: {np.min(bilstm_confidences):.4f}")
        msg.info(f"  Max: {np.max(bilstm_confidences):.4f}")
    
    if crf_confidences:
        msg.info(f"\nCRF Confidence:")
        msg.info(f"  Mean: {np.mean(crf_confidences):.4f}")
        msg.info(f"  Std: {np.std(crf_confidences):.4f}")
        msg.info(f"  Min: {np.min(crf_confidences):.4f}")
        msg.info(f"  Max: {np.max(crf_confidences):.4f}")
    
    # Entity-level statistics
    entity_tokens = [r for r in results if r['gold_label'] != "O"]
    if entity_tokens:
        entity_correct = sum(1 for r in entity_tokens if r['correct'])
        entity_accuracy = entity_correct / len(entity_tokens) if entity_tokens else 0
        msg.info(f"\nEntity Token Accuracy: {entity_correct}/{len(entity_tokens)} ({entity_accuracy:.2%})")
    
    msg.divider()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract confidence scores from NER model predictions")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (e.g., training/bilstm_crf_final/model-best)")
    parser.add_argument("--corpus", type=str, default="corpus/dev.spacy", help="Path to corpus file")
    parser.add_argument("--output", type=str, help="Output file path (default: metrics/{model_name}_confidence.json)")
    parser.add_argument("--format", type=str, choices=["json", "csv"], default="json", help="Output format")
    parser.add_argument("--limit", type=int, help="Limit number of documents to process")
    
    args = parser.parse_args()
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        msg.fail(f"Model not found: {model_path}")
        return
    
    # Register necessary factories before loading model
    register_factories_for_model(model_path)
    
    msg.info(f"Loading model from {model_path}")
    
    # Check if model config has include_static_vectors=true but vectors=null
    # This indicates a misconfigured model that can't be loaded
    config_path = model_path / "config.cfg"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
            # Check for problematic config: include_static_vectors=true with vectors=null
            has_include_static = 'include_static_vectors = true' in config_content
            import re
            paths_section = re.search(r'\[paths\](.*?)(?:\[|$)', config_content, re.DOTALL)
            vectors_is_null = False
            if paths_section:
                paths_content = paths_section.group(1)
                vectors_line = re.search(r'vectors\s*=\s*null', paths_content)
                vectors_is_null = vectors_line is not None
            
            if has_include_static and vectors_is_null:
                msg.fail(f"Model at {model_path} has include_static_vectors=true but vectors=null")
                msg.warn("This configuration is incompatible - the model expects vectors but none were saved.")
                msg.warn("To fix this, retrain the model with include_static_vectors=false in the config.")
                msg.info("Suggestion: Use configs/tok2vec.cfg with vectors=null and include_static_vectors=false")
                return
    
    nlp = spacy.load(model_path)
    model_name = model_path.parent.name  # e.g., "bilstm_crf_final"
    
    # Check if model uses static vectors
    uses_static_vectors = False
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
            import re
            paths_section = re.search(r'\[paths\](.*?)(?:\[|$)', config_content, re.DOTALL)
            if paths_section:
                paths_content = paths_section.group(1)
                vectors_line = re.search(r'vectors\s*=\s*(.+)', paths_content)
                if vectors_line:
                    vectors_value = vectors_line.group(1).strip()
                    uses_static_vectors = vectors_value != "null"
    
    if uses_static_vectors:
        vectors_shape = nlp.vocab.vectors.shape
        if vectors_shape[0] == 0:
            msg.warn(f"Model is configured to use static vectors but none are loaded (shape: {vectors_shape})")
            msg.fail("Please use a model with vectors loaded or one trained without static vectors.")
            return
        else:
            msg.good(f"Static vectors loaded: {vectors_shape}")
    else:
        msg.info("Model trains tok2vec from scratch (no static vectors)")
    
    # Get label mapping from the NER component
    ner = nlp.get_pipe("ner")
    idx_to_label = ner._idx_to_label if hasattr(ner, "_idx_to_label") else {}
    
    # Load corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        msg.fail(f"Corpus not found: {corpus_path}")
        return
    
    docs = load_corpus(corpus_path)
    
    # Process documents
    results = process_documents(nlp, docs, model_name, idx_to_label, limit=args.limit)
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("metrics")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{model_name}_confidence.{args.format}"
    
    save_results(results, output_path, format=args.format)


if __name__ == "__main__":
    main()
