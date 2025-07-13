"""
ARC Training Engine
LoRA + EWC trainer for continual learning with catastrophic forgetting prevention.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from pathlib import Path

from .config import ARCConfig
from .memory import MemorySystem
from .safety import SafetySystem


class ARCDataset(Dataset):
    """Dataset for ARC training."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format text based on item structure
        if "input" in item and "output" in item:
            text = f"Human: {item['input']}\nAssistant: {item['output']}"
        elif "text" in item:
            text = item["text"]
        else:
            text = str(item)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""
    
    def __init__(self, model, dataset_loader, lambda_ewc: float = 0.4):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_matrix = {}
        self.optimal_params = {}
        
        # Compute Fisher Information Matrix
        self._compute_fisher_matrix(dataset_loader)
    
    def _compute_fisher_matrix(self, dataset_loader):
        """Compute Fisher Information Matrix."""
        self.model.eval()
        
        # Initialize fisher matrix
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_matrix[name] = torch.zeros_like(param)
                self.optimal_params[name] = param.data.clone()
        
        # Compute gradients
        for batch in dataset_loader:
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(**batch)
            loss = outputs.loss
            
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_matrix[name] += param.grad.data ** 2
        
        # Normalize by dataset size
        dataset_size = len(dataset_loader.dataset)
        for name in self.fisher_matrix:
            self.fisher_matrix[name] /= dataset_size
    
    def penalty(self):
        """Compute EWC penalty."""
        penalty = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix:
                penalty += (self.fisher_matrix[name] * 
                           (param - self.optimal_params[name]) ** 2).sum()
        return self.lambda_ewc * penalty


class ARCTrainer:
    """Main ARC training class with LoRA and EWC."""
    
    def __init__(self, config: ARCConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.memory_system = MemorySystem(config.memory)
        self.safety_system = SafetySystem(config.safety)
        self.ewc = None
        
        # Training state
        self.training_history = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def _setup_logging(self):
        """Setup logging for training."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_model(self, base_model: str) -> Dict[str, Any]:
        """Initialize model with LoRA adapters."""
        self.logger.info(f"Initializing model: {base_model}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True
            )
            
            # Add LoRA adapters
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora.r,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                target_modules=self.config.lora.target_modules
            )
            
            self.model = get_peft_model(self.model, lora_config)
            
            if self.device.type != "cuda":
                self.model = self.model.to(self.device)
            
            # Update config
            context_length = getattr(self.model.config, 'max_position_embeddings', 2048)
            self.config.update_model_info(base_model, f"arc-{Path(base_model).name}", context_length)
            
            # Get model statistics
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            stats = {
                "base_model": base_model,
                "context_length": context_length,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "learning_percentage": (trainable_params / total_params) * 100,
                "device": str(self.device),
                "lora_rank": self.config.lora.r
            }
            
            self.logger.info(f"Model initialized successfully")
            self.logger.info(f"Trainable parameters: {trainable_params:,} ({stats['learning_percentage']:.2f}%)")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    def prepare_training_data(self, data_path: str) -> DataLoader:
        """Prepare training data from JSONL file."""
        data = []
        
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        dataset = ARCDataset(data, self.tokenizer, max_length=512)
        
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
    
    def train_on_pack(self, pack_path: str, update_ewc: bool = True) -> Dict[str, Any]:
        """Train on a teaching pack."""
        self.logger.info(f"Training on pack: {pack_path}")
        
        pack_dir = Path(pack_path)
        
        # Find training data files
        training_files = []
        for pattern in ["*.jsonl", "positive.jsonl", "negative.jsonl"]:
            training_files.extend(pack_dir.glob(pattern))
        
        if not training_files:
            raise FileNotFoundError(f"No training data found in {pack_path}")
        
        # Load all training data
        all_data = []
        for file_path in training_files:
            with open(file_path, 'r') as f:
                for line in f:
                    all_data.append(json.loads(line.strip()))
        
        # Create dataset and dataloader
        dataset = ARCDataset(all_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True)
        
        # Setup EWC if this isn't the first training
        if self.ewc is None and update_ewc:
            self.ewc = ElasticWeightConsolidation(
                self.model, dataloader, self.config.training.ewc_lambda
            )
        
        # Training setup
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Training loop
        self.model.train()
        total_loss = 0
        num_steps = 0
        
        for epoch in range(1):  # Single epoch for continual learning
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Add EWC penalty
                if self.ewc is not None:
                    loss += self.ewc.penalty()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_steps += 1
                
                if num_steps >= self.config.training.max_steps:
                    break
        
        avg_loss = total_loss / num_steps if num_steps > 0 else 0
        
        # Update EWC for future training
        if update_ewc:
            self.ewc = ElasticWeightConsolidation(
                self.model, dataloader, self.config.training.ewc_lambda
            )
        
        # Log training event
        training_event = {
            "timestamp": datetime.now().isoformat(),
            "pack": str(pack_path),
            "samples": len(all_data),
            "steps": num_steps,
            "avg_loss": avg_loss,
            "learning_rate": self.config.training.learning_rate
        }
        
        self.training_history.append(training_event)
        self._log_training_event(training_event)
        
        self.logger.info(f"Training completed. Steps: {num_steps}, Avg Loss: {avg_loss:.4f}")
        
        return training_event
    
    def _log_training_event(self, event: Dict[str, Any]):
        """Log training event to history file."""
        history_path = self.config.get_history_path()
        with open(history_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def save_model(self, output_path: str) -> str:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Initialize and train a model first.")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapters
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save ARC config
        config_path = output_path / "arc_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        self.logger.info(f"Model saved to {output_path}")
        return str(output_path)
    
    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate response with safety filtering."""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Apply safety filtering
        context = {"input": prompt, "type": "generation"}
        filtered_response = self.safety_system.filter_response(response, context)
        
        # Store in memory
        self.memory_system.store_interaction(prompt, filtered_response, context)
        
        return filtered_response
