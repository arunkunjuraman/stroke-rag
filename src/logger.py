import json
import os
from datetime import datetime
from typing import Any, Dict

class InteractionLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Use a daily log file
        self.log_file = os.path.join(self.log_dir, f"interactions_{datetime.now().strftime('%Y-%m-%d')}.jsonl")

    def log_interaction(self, state: Dict[str, Any]):
        """Logs the user question and LLM response."""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "question": state.get("question"),
            "answer": state.get("generation"),
            "is_cached": state.get("is_cached", False),
            "sources": []
        }

        for d in state.get("documents", []):
            if hasattr(d, "metadata"):
                metadata = d.metadata
            elif isinstance(d, dict):
                metadata = d.get("metadata", {})
            else:
                metadata = {}
            
            log_entry["sources"].append({
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "Unknown")
            })


        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"--- INFO: Interaction logged to {self.log_file} ---")
        except Exception as e:
            print(f"--- ERROR: Failed to log interaction: {str(e)} ---")

# Singleton instance
logger = InteractionLogger()
